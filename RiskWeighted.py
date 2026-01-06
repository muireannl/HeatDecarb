"""
Code to determine optimal method of decarbonising the residential heating sector with risk aversion.

Outputs are appended to scenario_summary.csv. Scenario name should be specified below after changing inputs
"""

import pandas as pd
import numpy as np
import pulp

# --- Load Data ---
df = pd.read_csv("inputs/stock.csv")
cost_df = pd.read_csv("inputs/costdata.csv")

#Import emissions
emissions_map = {}
for _, row in cost_df.iterrows():
    fuel = row['Fuel Category'].strip().lower()
    emissions_map[fuel] = row['Emissions Factor (kgCO2/MWh)']/1000

# --- Parameters ---
capital_costs = {
    'Apartment': 26713,
    'Terraced house': 52383,
    'Semi-D house': 59304,
    'Detached house': 63867
}

def annualise_capex(capex, rate=0.03, years=5):
    annuity_factor = (rate * (1 + rate) ** years) / ((1 + rate) ** years - 1)
    return capex * annuity_factor

disruption_cost = 26774.16
boiler_replacement_cost = 2500
include_disruption_cost = True
annualise_costs = False
retrofit = False
if retrofit:
    cop_heat_pump = 4
else:
    cop_heat_pump = 2.5
emissions_cap = 4797600  # tonnes CO2
biofuel_caps = {"hvo": 100000, "lpg": 100000, "biolpg": 25000}

scenario_name = "Optimised_Dis_NoAnn_MedCap_Retro_HP_GB"

# Generate random prices
# Set seed for reproducibility
np.random.seed(42)

# Step 1: Define log-space mean vector and covariance matrix
mean_prices = {
    'Electric heating': 370.00,
    'Gas boiler': 122.200,
    'Oil boiler': 90.4,
    'LPG': 162.4,
    'BioLPG': 210.40,
    'HVO': 178.40
}

# Extract log-space means (μ_log) from the price vector
mu_log = np.log(list(mean_prices.values()))

# Extract covariance matrix (Σ_log) from previous step
# (values taken from last known working matrix and scaled appropriately)
# Covariance matrix should be symmetric and positive definite
cov_log = np.array([
    [0.045129,  0.029804,  0.017503,  0.015753,  0.015753,  0.015753],
    [0.029804,  0.065536,  0.035147,  0.031632,  0.031632,  0.031632],
    [0.017503,  0.035147,  0.078167,  0.070350,  0.070350,  0.070350],
    [0.015753,  0.031632,  0.070350,  0.070350,  0.070350,  0.070350],
    [0.015753,  0.031632,  0.070350,  0.070350,  0.070350,  0.070350],
    [0.015753,  0.031632,  0.070350,  0.070350,  0.070350,  0.070350],
])

# Step 2: Function to draw N fuel price scenarios
def draw_fuel_price_scenarios(n=1):
    samples_log = np.random.multivariate_normal(mu_log, cov_log, size=n)
    samples_price = np.exp(samples_log)
    df_samples = pd.DataFrame(samples_price, columns=mean_prices.keys())
    return df_samples

#draw a full matrix of fuel price scenarios for use within the optimisation loop
n_scenarios = 10000
fuel_price_scenarios = draw_fuel_price_scenarios(n_scenarios)

# Add solid boiler as a fixed value for all scenarios
fuel_price_scenarios['solid boiler'] = 70.1

# Standardize the column names
fuel_price_scenarios.columns = [col.strip().lower() for col in fuel_price_scenarios.columns]

#fuel_price_scenarios.to_csv("outputs/fuel_prices_generated.csv", index=False)

fuel_price_scenarios = pd.read_csv("outputs/fuel_prices_generated.csv")


# --- Preprocess Main Data ---
df['Fuel Category'] = df['Fuel Category'].str.lower()
df['Key'] = df['BER Group'] + ":" + df['Dwelling Type'] + ":" + df['Fuel Category']
df.set_index('Key', inplace=True)

# --- Decision Variables ---
prob = pulp.LpProblem("HeatDecarbonisation", pulp.LpMinimize)
decisions = {}

# For each group: allow stay, retrofit to gas/heat pump, or biofuel switch
for key, row in df.iterrows():
    ber, dwelling_type, fuel = key.split(":")
    dwellings = row['Dwellings']

    # Stay as-is
    var_stay = pulp.LpVariable(f"stay_{key}", lowBound=0, upBound=dwellings)
    decisions[(key, 'stay')] = var_stay

    # Retrofit options (only for BER < A1-B2)
    if ber != "A1-B2":
        for new_fuel in ["gas boiler", "heatpump"]:
            tag = f"retrofit_{new_fuel}_{key}"
            var = pulp.LpVariable(tag, lowBound=0, upBound=dwellings)
            decisions[(key, new_fuel)] = var

    # Biofuel options (only for oil boilers)
    if fuel == "oil boiler" and ber != "A1-B2":
        for bio in ["lpg", "biolpg", "hvo"]:
            tag = f"switch_{bio}_{key}"
            var = pulp.LpVariable(tag, lowBound=0, upBound=dwellings)
            decisions[(key, bio)] = var

# --- Constraints ---

# 1. Must allocate all dwellings (not just stay within stock)
for key, row in df.iterrows():
    total = pulp.lpSum([var for (k, _), var in decisions.items() if k == key])
    prob += total == row['Dwellings'], f"must_allocate_{key}"

# 2. Maintain dwelling type counts
for dwelling_type in df['Dwelling Type'].unique():
    original_total = df[df['Dwelling Type'] == dwelling_type]['Dwellings'].sum()
    new_total = pulp.lpSum([var for (k, action), var in decisions.items() if dwelling_type in k])
    prob += new_total == original_total, f"dwelling_type_consistency_{dwelling_type}"

# 3. Emissions constraint
emissions = []
for (key, action), var in decisions.items():
    ber, dwelling_type, fuel = key.split(":")

    if retrofit:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip().lower(), 0)
        elif action == 'gas boiler':
            ei = df[(df['Fuel Category'] == 'gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map['gas boiler']
        elif action == 'heatpump':
            ei = df[(df['Fuel Category'] == 'electric heating') & (df['BER Group'] == 'A1-B2')][
                     'EnergyIntensity'].mean() / cop_heat_pump
            ef = emissions_map['electric heating']
        elif action in ['lpg', 'biolpg', 'hvo']:
            ei = df[(df['Fuel Category'] == action) & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map.get(action, 0)
        else:
            ei = 0
            ef = 0
    else:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip().lower(), 0)
        elif action == 'gas boiler':
            ei = df[(df['Fuel Category'] == 'gas boiler')]['EnergyIntensity'].mean()
            ef = emissions_map['gas boiler']
        elif action == 'heatpump':
            ei = df[(df['Fuel Category'] == 'electric heating')][
                     'EnergyIntensity'].mean() / cop_heat_pump
            ef = emissions_map['electric heating']
        elif action in ['lpg', 'biolpg', 'hvo']:
            ei = df[(df['Fuel Category'] == action)]['EnergyIntensity'].mean()
            ef = emissions_map.get(action, 0)
        else:
            ei = 0
            ef = 0

    print(ei, ef)

    emissions.append(ef * ei * var)

prob += pulp.lpSum(emissions) <= emissions_cap, "carbon_cap"

# 4. Biofuel switching caps
for bio, cap in biofuel_caps.items():
    total_switch = pulp.lpSum([var for (k, act), var in decisions.items() if act == bio])
    prob += total_switch <= cap, f"switch_cap_{bio}"

# 5. Cap on total switches from oil boiler to gas boiler
oil_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.lower().endswith('oil boiler')
])

prob += oil_switch <= 83537, "OilToGasCap"

#6. Cap on total switches from solid boiler to gas boiler
solid_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.lower().endswith('solid boiler')
])

prob += oil_switch <= 10272, "SolidToGasCap"

#7. Cap on total switches from electricity to gas boiler
solid_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.lower().endswith('electric heating')
])

prob += oil_switch <= 11374, "ElecToGasCap"

# --- Objective Function: Total Cost ---
def compute_total_cost(cost_map, decisions, df, retrofit):
    total_cost = []
    for (key, action), var in decisions.items():
        ber, dwelling_type, fuel = key.split(":")

        if retrofit:
            if action == 'stay':
                ei = df.loc[key]['EnergyIntensity']
                unit_cost = cost_map.get(fuel.strip().lower(), None)
                if pd.isna(ei) or unit_cost is None:
                    raise ValueError(f"Missing stay cost for {key}: EI={ei}, unit_cost={unit_cost}")
                total_cost.append(unit_cost * ei * var)
            elif action == 'gas boiler':
                fuel_cost = cost_map['gas boiler']
                ei = df[(df['Fuel Category'] == 'gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()

                cap = capital_costs[dwelling_type]
                annualised_cap = annualise_capex(cap)
                cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
                if include_disruption_cost:
                    cost += disruption_cost  # still one-off unless you want it annualised too
                total_cost.append(cost * var)
            elif action == 'heatpump':
                fuel_cost = cost_map['electric heating']
                ei = df[(df['Fuel Category'] == 'electric heating') & (df['BER Group'] == 'A1-B2')][
                     'EnergyIntensity'].mean() / cop_heat_pump
                cap = capital_costs[dwelling_type]
                annualised_cap = annualise_capex(cap)
                cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
                if include_disruption_cost:
                    cost += disruption_cost  # still treated as a one-off
                total_cost.append(cost * var)
            elif action in ['lpg', 'biolpg', 'hvo']:
                fuel_cost = cost_map[action]
                ei = df[(df['Fuel Category'] == action)]['EnergyIntensity'].mean()
                cost = fuel_cost * ei + boiler_replacement_cost
                total_cost.append(cost * var)
        else:
            if action == 'stay':
                ei = df.loc[key]['EnergyIntensity']
                unit_cost = cost_map.get(fuel.strip().lower(), None)
                if pd.isna(ei) or unit_cost is None:
                    raise ValueError(f"Missing stay cost for {key}: EI={ei}, unit_cost={unit_cost}")
                total_cost.append(unit_cost * ei * var)
            elif action == 'gas boiler':
                fuel_cost = cost_map['gas boiler']
                ei = df[(df['Fuel Category'] == 'gas boiler')]['EnergyIntensity'].mean()
                cap = 3500
                annualised_cap = annualise_capex(cap)
                cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
                if include_disruption_cost:
                    cost += disruption_cost
                total_cost.append(cost * var)
            elif action == 'heatpump':
                fuel_cost = cost_map['electric heating']
                ei = df[(df['Fuel Category'] == 'electric heating')]['EnergyIntensity'].mean() / cop_heat_pump
                cap = 15000
                annualised_cap = annualise_capex(cap)
                cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
                if include_disruption_cost:
                    cost += disruption_cost  # still treated as a one-off
                total_cost.append(cost * var)
            elif action in ['lpg', 'biolpg', 'hvo']:
                fuel_cost = cost_map[action]
                ei = df[(df['Fuel Category'] == action)]['EnergyIntensity'].mean()
                cost = fuel_cost * ei + boiler_replacement_cost
                total_cost.append(cost * var)

    return pulp.lpSum(total_cost)


#Now build the objective function and solve

scenario_costs = []
for i in range(n_scenarios):
    sampled_prices = fuel_price_scenarios.iloc[i].to_dict()
    cost_map = {k.strip().lower(): v for k, v in sampled_prices.items()}
    scenario_cost = compute_total_cost(cost_map, decisions, df, retrofit)
    scenario_costs.append(scenario_cost)


# Risk aversion parameters
# Define risk parameters
beta = 0.95
lambda_param = 1.0

# Define auxiliary variables
z = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')  # Value-at-Risk
u = [pulp.LpVariable(f"u_{i}", lowBound=0, cat='Continuous') for i in range(n_scenarios)]  # Deviations above VaR

# CVaR constraints
for i in range(n_scenarios):
    prob += u[i] >= scenario_costs[i] - z, f"CVaR_exceedance_{i}"

# Objective function: Expected Cost + λ * CVaR
expected_cost = (1 / n_scenarios) * pulp.lpSum(scenario_costs)
cvar_term = z + (1 / ((1 - beta) * n_scenarios)) * pulp.lpSum(u)
prob += expected_cost + lambda_param * cvar_term, "TotalCostPlusCVaR"

# Solve the problem
prob.solve()

# Check results
print("Status:", pulp.LpStatus[prob.status])
print("Objective value:", pulp.value(prob.objective))

print(f"Total cost: €{pulp.value(prob.objective):,.2f}")
print(f"Total emissions: {pulp.value(pulp.lpSum(emissions)):.2f} tonnes CO2")

#Process and print results

results = []

from collections import defaultdict

action_counts = defaultdict(float)

for (key, action), var in decisions.items():
    if pulp.value(var) is not None:
        action_counts[action] += pulp.value(var)

# Print results
for action, total in action_counts.items():
    print(f"{action}: {total:.0f} dwellings")

# Create a DataFrame from the action_counts dictionary
action_df = pd.DataFrame(list(action_counts.items()), columns=["Action", "Total Dwellings"])

# Round the totals
action_df["Total Dwellings"] = action_df["Total Dwellings"].round()

# Create a filename with the lambda value
filename = f"outputs/action_counts_lambda_{lambda_param:.2f}.csv"

# Save to CSV
action_df.to_csv(filename, index=False)

print(f"Saved action counts to {filename}")

# --- Write final portfolio to CSV ---
portfolio = []

for (key, action), var in decisions.items():
    ber, dwelling_type, fuel = key.split(":")
    dwellings = var.varValue or 0
    if dwellings > 0:
        # Determine Energy Intensity and Emissions Factor
        if action == "stay":
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip().lower(), 0)
            cost_per_mwh = cost_map.get(fuel.strip().lower(), 0)
            capex = 0
        elif action == "gas boiler":
            ei = df[(df['Fuel Category'] == 'gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map['gas boiler']
            cost_per_mwh = cost_map['gas boiler']
            capex = capital_costs[dwelling_type] + (disruption_cost if include_disruption_cost else 0)
        elif action == "heatpump":
            ei = df[(df['Fuel Category'] == 'electric heating') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean() / cop_heat_pump
            ef = emissions_map['electric heating']
            cost_per_mwh = cost_map['electric heating']
            capex = capital_costs[dwelling_type] + (disruption_cost if include_disruption_cost else 0)
        elif action in ['lpg', 'biolpg', 'hvo']:
            ei = df[(df['Fuel Category'] == action) & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map[action]
            cost_per_mwh = cost_map[action]
            capex = boiler_replacement_cost
        else:
            ei, ef, cost_per_mwh, capex = 0, 0, 0, 0

        opex = ei * cost_per_mwh
        total_cost = (opex + capex) * dwellings
        total_emissions = ef * ei * dwellings

        portfolio.append({
            'BER Group': ber,
            'Dwelling Type': dwelling_type,
            'Original Fuel': fuel,
            'Action': action,
            'Dwellings': dwellings,
            'EnergyIntensity': ei,
            'EmissionFactor': ef,
            'OPEX €/dwelling': opex,
            'CAPEX €/dwelling': capex,
            'Total Cost (€)': total_cost,
            'Total Emissions (tCO2)': total_emissions
        })

portfolio_df = pd.DataFrame(portfolio)

# Create a filename with the lambda value
filename = f"outputs/final_portfolio_lambda_{lambda_param:.2f}.csv"

# Save to CSV
portfolio_df.to_csv(filename, index=False)
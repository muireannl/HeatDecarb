"""
Code to determine optimal method of decarbonising the residential heating sector.

Outputs are appended to scenario_summary.csv. Scenario name should be specified below after changing inputs
"""

import pandas as pd
import pulp
import os

# --- Load Data ---
df = pd.read_csv("inputs/stock.csv")
cost_df = pd.read_csv("inputs/costdata.csv")

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
include_disruption_cost = False
annualise_costs = False
retrofit = False
if retrofit:
    cop_heat_pump = 4
else:
    cop_heat_pump = 2.8

emissions_cap = 4797600  # tonnes CO2
biofuel_caps = {"HVO": 10000, "LPG": 10000, "BioLPG": 2500}


# --- Preprocess Costs ---
cost_map = {}
emissions_map = {}
for _, row in cost_df.iterrows():
    fuel = row['Fuel Category']
    cost_map[fuel] = row['euro/MWh (Excl. Carbon Tax)'] + row['Carbon Tax (euro/MWh)']
    emissions_map[fuel] = row['Emissions Factor (kgCO2/MWh)']/1000

# --- Preprocess Main Data ---
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
    if fuel == "Oil boiler" and ber != "A1-B2":
        for bio in ["LPG", "BioLPG", "HVO"]:
            tag = f"switch_{bio}_{key}"
            var = pulp.LpVariable(tag, lowBound=0, upBound=dwellings)
            decisions[(key, bio)] = var

# --- Constraints ---

# 1. Must allocate all dwellings (including to "stay")
for key, row in df.iterrows():
    total = pulp.lpSum([var for (k, _), var in decisions.items() if k == key])
    prob += total == row['Dwellings'], f"must_allocate_{key}"

# 2. Maintain dwelling type counts
for dwelling_type in df['Dwelling Type'].unique():
    original_total = df[df['Dwelling Type'] == dwelling_type]['Dwellings'].sum()
    new_total = pulp.lpSum([var for (k, action), var in decisions.items() if k.split(":")[1] == dwelling_type
])
    prob += new_total == original_total, f"dwelling_type_consistency_{dwelling_type}"

# --- Emissions constraint ---
emissions = []
for (key, action), var in decisions.items():
    ber, dwelling_type, fuel = key.split(":")

    if retrofit:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip(), 0)
        elif action == 'gas boiler':
            ei = df[(df['Fuel Category'] == 'Gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map.get('Gas boiler', 0)
        elif action == 'heatpump':
            ei = (
                df[(df['Fuel Category'] == 'Electric heating') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity']
                .mean()
            ) / cop_heat_pump
            ef = emissions_map.get('Electric heating', 0)
        elif action in ['LPG', 'BioLPG', 'HVO']:
            ei = df[(df['Fuel Category'] == action) & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            ef = emissions_map.get(action, 0)
        else:
            ei = 0
            ef = 0
    else:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip(), 0)
        elif action == 'gas boiler':
            ei = df[df['Fuel Category'] == 'Gas boiler']['EnergyIntensity'].mean()
            ef = emissions_map.get('Gas boiler', 0)
        elif action == 'heatpump':
            ei = df[df['Fuel Category'] == 'Electric heating']['EnergyIntensity'].mean() / cop_heat_pump
            ef = emissions_map.get('Electric heating', 0)
        elif action in ['LPG', 'BioLPG', 'HVO']:
            ei = df[df['Fuel Category'] == action]['EnergyIntensity'].mean()
            ef = emissions_map.get(action, 0)
        else:
            ei = 0
            ef = 0

    # tiny guard in case a .mean() produced NaN (empty slice)
    if pd.isna(ei):
        ei = 0

    # print(ei, ef)

    emissions.append(ef * ei * var)

prob += pulp.lpSum(emissions) <= emissions_cap, "carbon_cap"

# 4. Biofuel switching caps
for bio, cap in biofuel_caps.items():
    total_switch = pulp.lpSum([var for (k, act), var in decisions.items() if act == bio])
    prob += total_switch <= cap, f"switch_cap_{bio}"

# 5. Cap on total switches from oil boiler to gas boiler
oil_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.split(":")[2] == 'Oil boiler'
])
prob += oil_switch <= 83537, "OilToGasCap"

# 6. Cap on total switches from solid boiler to gas boiler
solid_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.split(":")[2] == 'Solid boiler'
])
prob += solid_switch <= 10272, "SolidToGasCap"

# 7. Cap on total switches from electricity to gas boiler
elec_switch = pulp.lpSum([
    var for (key, action), var in decisions.items()
    if action == 'gas boiler' and key.split(":")[2] == 'Electric heating'
])
prob += elec_switch <= 11374, "ElecToGasCap"

# --- Objective Function: Total Cost ---
total_cost = []
for (key, action), var in decisions.items():
    ber, dwelling_type, fuel = key.split(":")

    if retrofit:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            unit_cost = cost_map.get(fuel.strip(), None)
            if pd.isna(ei) or unit_cost is None:
                raise ValueError(f"Missing stay cost for {key}: EI={ei}, unit_cost={unit_cost}")
            total_cost.append(unit_cost * ei * var)
        elif action == 'gas boiler':
            fuel_cost = cost_map['Gas boiler']
            ei = df[(df['Fuel Category'] == 'Gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            cap = capital_costs[dwelling_type]
            annualised_cap = annualise_capex(cap)
            cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
            if include_disruption_cost:
                cost += disruption_cost
            total_cost.append(cost * var)
        elif action == 'heatpump':
            fuel_cost = cost_map['Electric heating']
            ei = df[(df['Fuel Category'] == 'Electric heating') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean() / cop_heat_pump
            cap = capital_costs[dwelling_type]
            annualised_cap = annualise_capex(cap)
            cost = fuel_cost * ei + (annualised_cap if annualise_costs else cap)
            if include_disruption_cost:
                cost += disruption_cost
            total_cost.append(cost * var)
        elif action in ['LPG', 'BioLPG', 'HVO']:
            fuel_cost = cost_map[action]
            ei = df[(df['Fuel Category'] == action)]['EnergyIntensity'].mean()
            cost = fuel_cost * ei + boiler_replacement_cost
            total_cost.append(cost * var)

    else:
        if action == 'stay':
            ei = df.loc[key]['EnergyIntensity']
            unit_cost = cost_map.get(fuel.strip(), None)
            if pd.isna(ei) or unit_cost is None:
                raise ValueError(f"Missing stay cost for {key}: EI={ei}, unit_cost={unit_cost}")
            total_cost.append(unit_cost * ei * var)

        elif action == 'gas boiler':
            fuel_cost = cost_map['Gas boiler']
            ei = df[df['Fuel Category'] == 'Gas boiler']['EnergyIntensity'].mean()
            cap = 3500
            annualised_cap = annualise_capex(cap)
            cost = fuel_cost * ei + (annualise_costs and annualised_cap or cap)
            if include_disruption_cost:
                cost += disruption_cost
            total_cost.append(cost * var)

        elif action == 'heatpump':
            fuel_cost = cost_map['Electric heating']
            ei = df[df['Fuel Category'] == 'Electric heating']['EnergyIntensity'].mean() / cop_heat_pump
            cap = 15000
            annualised_cap = annualise_capex(cap)
            cost = fuel_cost * ei + (annualise_costs and annualised_cap or cap)
            if include_disruption_cost:
                cost += disruption_cost
            total_cost.append(cost * var)

        elif action in ['LPG', 'BioLPG', 'HVO']:
            fuel_cost = cost_map[action]
            ei = df[df['Fuel Category'] == action]['EnergyIntensity'].mean()
            cost = fuel_cost * ei + boiler_replacement_cost
            total_cost.append(cost * var)

prob += pulp.lpSum(total_cost), "TotalCost"

# --- Solve ---
prob.solve()
print(f"Status: {pulp.LpStatus[prob.status]}")
print(f"Total cost: €{pulp.value(prob.objective):,.2f}")
print(f"Total emissions: {pulp.value(pulp.lpSum(emissions)):.2f} tonnes CO2")

results = []

for (key, action), var in decisions.items():
    val = var.varValue
    if val is not None and val > 0:
        ber, dwelling_type, fuel = key.split(":")
        results.append({
            "BER Group": ber,
            "Dwelling Type": dwelling_type,
            "Fuel Category": fuel,
            "Action": action,
            "Dwellings": val
        })

for bio, cap in biofuel_caps.items():
    assigned = sum(var.varValue for (k, act), var in decisions.items() if act == bio)
    if assigned > cap + 1e-3:
        print(f"Biofuel cap violated for {bio}: assigned {assigned:.2f}, cap {cap}")

results_df = pd.DataFrame(results)
print(results_df)

#results_df.to_csv("outputs/final_portfolio.csv", index=False)

# --- Write final portfolio to CSV ---
portfolio = []

for (key, action), var in decisions.items():
    ber, dwelling_type, fuel = key.split(":")
    dwellings = var.varValue or 0
    if dwellings > 0:
        # Determine Energy Intensity and Emissions Factor
        if action == "stay":
            ei = df.loc[key]['EnergyIntensity']
            ef = emissions_map.get(fuel.strip(), 0)
            cost_per_mwh = cost_map.get(fuel.strip(), 0)
            capex = 0

        elif action == "gas boiler":
            if retrofit:
                ei = df[(df['Fuel Category'] == 'Gas boiler') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
                capex = capital_costs[dwelling_type] + (disruption_cost if include_disruption_cost else 0)
            else:
                ei = df[df['Fuel Category'] == 'Gas boiler']['EnergyIntensity'].mean()
                capex = 3500 + (disruption_cost if include_disruption_cost else 0)
            ef = emissions_map.get('Gas boiler', 0)
            cost_per_mwh = cost_map['Gas boiler']

        elif action == "heatpump":
            if retrofit:
                base_ei = df[(df['Fuel Category'] == 'Electric heating') & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
                capex = capital_costs[dwelling_type] + (disruption_cost if include_disruption_cost else 0)
            else:
                base_ei = df[df['Fuel Category'] == 'Electric heating']['EnergyIntensity'].mean()
                capex = 15000 + (disruption_cost if include_disruption_cost else 0)
            ei = base_ei / cop_heat_pump
            ef = emissions_map.get('Electric heating', 0)
            cost_per_mwh = cost_map['Electric heating']

        elif action in ['LPG', 'BioLPG', 'HVO']:
            if retrofit:
                ei = df[(df['Fuel Category'] == action) & (df['BER Group'] == 'A1-B2')]['EnergyIntensity'].mean()
            else:
                ei = df[df['Fuel Category'] == action]['EnergyIntensity'].mean()
            ef = emissions_map.get(action, 0)
            cost_per_mwh = cost_map[action]
            capex = boiler_replacement_cost

        else:
            ei, ef, cost_per_mwh, capex = 0, 0, 0, 0

        if pd.isna(ei):
            ei = 0

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
            'EmissionFactor_tCO2_per_MWh': ef,
            'OPEX €/dwelling': opex,
            'CAPEX €/dwelling': capex,
            'Total Cost (€)': total_cost,
            'Total Emissions (tCO2)': total_emissions
        })

# Save to CSV
scenario_name = f"deterministic_low_Dis_{include_disruption_cost:.2f}_Ann_{annualise_costs:.2f}_retro_{retrofit:.2f}"

portfolio_df = pd.DataFrame(portfolio)
outfile = f"outputs/{scenario_name}.csv"
portfolio_df.to_csv(outfile, index=False)
print(f"Final portfolio saved to '{outfile}'")

# Print summary
print(f"Total cost: €{pulp.value(prob.objective):,.2f}")
print(f"Total emissions: {pulp.value(pulp.lpSum(emissions)):.2f} tonnes CO2")
print(f"Final portfolio saved to '{outfile}'")

# --- Calculate total boiler replacement cost ---
biofuel_actions = ['LPG', 'BioLPG', 'HVO']
boiler_replacement_total = portfolio_df[
    portfolio_df['Action'].isin(biofuel_actions)
]['Dwellings'].sum() * boiler_replacement_cost

print(f"Total boiler replacement cost: €{boiler_replacement_total:,.2f}")

# Compute totals
opex_cost = portfolio_df['OPEX €/dwelling'].multiply(portfolio_df['Dwellings']).sum()
carbon_cost = portfolio_df['Total Emissions (tCO2)'].sum() * 100  # €/t = 100
fuel_cost = opex_cost - carbon_cost
capex = portfolio_df[portfolio_df['Action'].isin(['gas boiler', 'heatpump'])]['CAPEX €/dwelling']\
        .multiply(portfolio_df['Dwellings']).sum()

# Sanity check for consistent carbon pricing
diff = abs((fuel_cost + carbon_cost) - opex_cost)
if diff > 1e-6:
    print(f"Warning: OPEX split mismatch = {diff:.6f} € (check carbon price alignment)")

# Disruption already included in CAPEX €/dwelling earlier
disruption = 0.0

boiler_cost = portfolio_df[portfolio_df['Action'].isin(['HVO', 'LPG', 'BioLPG'])]['Dwellings'].sum() * boiler_replacement_cost
biomethane_cost = 0
emissions_total = portfolio_df['Total Emissions (tCO2)'].sum()
total_cost = fuel_cost + carbon_cost + capex + disruption + boiler_cost + biomethane_cost

# Count total dwellings per action (guard None)
action_counts = {}
for (key, action), var in decisions.items():
    action_counts[action] = action_counts.get(action, 0) + (var.varValue or 0)

# Prepare row to append (counts rounded to integers)
row = {
    "Scenario": scenario_name,
    "Fuel Cost (€)": round(fuel_cost),
    "Carbon Cost (€)": round(carbon_cost, 1),
    "CAPEX (€)": round(capex),
    "Disruption Cost (€)": round(disruption),
    "Boiler Replacement (€)": round(boiler_cost),
    "Biomethane Cost (€)": round(biomethane_cost),
    "Total Emissions (tCO2)": round(emissions_total, 3),
    "Total Cost (€)": round(total_cost)
}
for action in ['stay', 'heatpump', 'gas boiler', 'HVO', 'LPG', 'BioLPG']:
    row[f"# {action}"] = int(round(action_counts.get(action, 0)))

summary_file = "outputs/summary_deterministic_low.csv"

os.makedirs("outputs", exist_ok=True)

if os.path.exists(summary_file):
    df_summary = pd.read_csv(summary_file)
    df_summary = pd.concat([df_summary, pd.DataFrame([row])], ignore_index=True)
else:
    df_summary = pd.DataFrame([row])

df_summary.to_csv(summary_file, index=False)
print(f"Results appended to {summary_file}")

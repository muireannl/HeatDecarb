"""
Code to calculate the subsidies and costs bourne by different players
"""

import pandas as pd
import os

# Define the file paths (adjust as needed)
scenario_files = {
    'Baseline': r'C:\Users\MLynch\OneDrive - Economic and Social Research Institute\DCC\outputs\Baseline_detailed.csv',
    'MeetTarget': r'C:\Users\MLynch\OneDrive - Economic and Social Research Institute\DCC\outputs\MeetTarget_detailed.csv',
    'MissTarget': r'C:\Users\MLynch\OneDrive - Economic and Social Research Institute\DCC\outputs\MissTarget_detailed.csv',
    'CompGas': r'C:\Users\MLynch\OneDrive - Economic and Social Research Institute\DCC\outputs\CompGas_detailed.csv',
    'CompOil': r'C:\Users\MLynch\OneDrive - Economic and Social Research Institute\DCC\outputs\CompOil_detailed.csv',
}

# Read all files into a dictionary of DataFrames
scenario_data = {}
for name, filepath in scenario_files.items():
    scenario_data[name] = pd.read_csv(filepath)

print("Total dwellings in Baseline:", scenario_data['Baseline']['Dwellings'].sum())
print("Total MWh in Baseline:", scenario_data['Baseline']['Total MWh'].sum())


grant_ratios = {
    'Apartment': {'Grant': 8500, 'Homeowner': 18213},
    'Terraced house': {'Grant': 19800, 'Homeowner': 32048},
    'Semi-D house': {'Grant': 22000, 'Homeowner': 37800},
    'Detached house': {'Grant': 23700, 'Homeowner': 38058}
}

# --- 3. Fuel VAT & excise assumptions ---
fuel_tax_rates = {
    'Electric heating': {'VAT': 0.09, 'Excise': 0.00},
    'Heat pump': {'VAT': 0.09, 'Excise': 0.00},
    'Gas boiler': {'VAT': 0.09, 'Excise': 0.00},
    'Oil boiler': {'VAT': 0.135, 'Excise': 0.00},
    'LPG': {'VAT': 0.135, 'Excise': 14.60},
    'BioLPG': {'VAT': 0.135, 'Excise': 0.00},
    'HVO': {'VAT': 0.135, 'Excise': 0.00},
    'Solid boiler (blended)': {'VAT': 0.135, 'Excise': 10.35}
}

cost = pd.read_csv("inputs/costdata.csv")

def get_price_per_mwh(fuel: str, cost_df: pd.DataFrame) -> float:
    f = str(fuel).strip()
    row = cost_df.loc[cost_df["Fuel Category"] == f]
    if row.empty:
        raise ValueError(f"Fuel '{fuel}' not found in cost data.")
    return float(row["euro/MWh (Excl. Carbon Tax)"].iloc[0])

boiler_replacement_cost = 2500.0

results = {}

# Process each scenario
for scenario, filepath in scenario_files.items():
    df = pd.read_csv(filepath)

    # Add ratio of grant to total
    df['Grant_share'] = df['Dwelling Type'].map(lambda x: grant_ratios[x]['Grant'] /
                                                        (grant_ratios[x]['Grant'] + grant_ratios[x]['Homeowner']))
    df['Homeowner_share'] = 1 - df['Grant_share']

    # Split CAPEX
    df['Grant_total'] = df['Capital Cost (€)'] * df['Grant_share']
    df['Homeowner_total'] = df['Capital Cost (€)'] * df['Homeowner_share']

    # Sum totals
    total_grant = df['Grant_total'].sum()
    total_homeowner = df['Homeowner_total'].sum()
    total_capex = df['Capital Cost (€)'].sum()

    print(f"{scenario}:")
    print(f"  Grant total:     €{total_grant:,.0f}")
    print(f"  Homeowner total: €{total_homeowner:,.0f}")
    print(f"  Total CAPEX:     €{total_capex:,.0f}")
    print(f"  Matches:         {abs((total_grant + total_homeowner) - total_capex) < 1e-2}")
    print()


    # ✅ Carbon tax: use full dataset
    total_carbon_tax = df['Carbon Tax (euro)'].sum()

    # ✅ Fuel costs: also use full dataset
    fuel_cost = 0
    vat_total = 0
    excise_total = 0

    for _, row in df.iterrows():
        fuel = str(row['Fuel Category']).strip()
        if pd.isna(fuel) or fuel not in fuel_tax_rates:
            print(f"Skipping fuel type: {fuel}")
            continue


        energy_mwh = row['Total MWh']
        expenditure = row['Fuel Expenditure (euro)']
        price_per_mwh = get_price_per_mwh(fuel, cost)
        vat = fuel_tax_rates[fuel]['VAT']
        excise = fuel_tax_rates[fuel]['Excise']

        if scenario == "Baseline":
            print({
                'fuel': fuel,
                'energy_mwh': energy_mwh,
                'price': price_per_mwh,
                'vat': vat,
                'excise': excise
            })

        #    print(f"fuel_component = {energy_mwh * (price_per_mwh / (1 + vat) - excise)}")

        base_price_with_excise = price_per_mwh / (1 + vat)
        base_price_excl_excise = base_price_with_excise / (1 + excise)

        fuel_component = energy_mwh * base_price_excl_excise
        excise_component = energy_mwh * excise
        vat_component = energy_mwh * base_price_with_excise * vat

        fuel_cost += fuel_component
        excise_total += excise_component
        vat_total += vat_component

    # --- Boiler replacement CAPEX for oil-to-biofuel switches ---
    boiler_switch_capex = 0.0

    biofuels = ["HVO", "LPG", "BioLPG"]

    for _, row in df.iterrows():
        if row.get('Retrofitted', False):
            continue  # boiler replacement not needed if fully retrofitted

        fuel = str(row['Fuel Category']).strip()
        if fuel in biofuels:
            num_dwellings = row['Dwellings']
            boiler_switch_capex += num_dwellings * boiler_replacement_cost

    # --- Store Results ---
    results[scenario] = {
        'Grant': total_grant,
        'Homeowner CAPEX': total_homeowner,
        'Boiler CAPEX': boiler_switch_capex,
        'Total CAPEX': total_grant + total_homeowner + boiler_switch_capex,
        'Carbon Tax': total_carbon_tax,
        'Fuel Cost (excl. tax)': fuel_cost,
        'VAT': vat_total,
        'Excise': excise_total,
        'Fuel w VATExcise': fuel_cost + vat_total + excise_total
    }

# --- Export ---
summary_df = pd.DataFrame.from_dict(results, orient='index')
summary_df = summary_df.round(0)
summary_df.to_csv('subsidy_costs_by_scenario.csv')
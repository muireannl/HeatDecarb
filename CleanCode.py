
import os
import pandas as pd

# Load
stock = pd.read_csv("inputs/stock.csv")
cost = pd.read_csv("inputs/costdata.csv")

print (cost)

wtp_df = pd.read_csv("WTPdata.csv")
median_disruption_cost = abs(wtp_df["wtpDisMajor"][wtp_df["wtpDisMajor"] < 0].median()*1000)

capital_costs = {
    'Apartment': 26713,
    'Terraced house': 52383,
    'Semi-D house': 59304,
    'Detached house': 63867
}

boiler_upgrade_capex = 2500.0

heatpump_capex = 14000.00

scenarios = {
    'Baseline': {
        'Retrofit_B2_000s': 000,
        'HeatPump_000s': 000,
        'DistrictHeating_TWh': 0,
        'HydrogenBlend': 0.0,
        'Biomethane_TWh': 0.0,
        'HVO': 0,
        'LPG': 0,
        'BioLPG': 0
    },
    'MeetTarget': {
        'Retrofit_B2_000s': 500,
        'HeatPump_000s': 400,
        'DistrictHeating_TWh': 2.7,
        'HydrogenBlend': 0.0,
        'Biomethane_TWh': 0.0,
        'HVO': 0,
        'LPG': 0,
        'BioLPG': 0
    },
    'MissTarget': {
        'Retrofit_B2_000s': 300,
        'HeatPump_000s': 200,
        'DistrictHeating_TWh': 2.7,
        'HydrogenBlend': 0.0,
        'Biomethane_TWh': 0.0,
        'HVO': 0,
        'LPG': 0,
        'BioLPG': 0
    },
    'CompGas': {
        'Retrofit_B2_000s': 300,
        'HeatPump_000s': 200,
        'DistrictHeating_TWh': 2.7,
        'HydrogenBlend': 0.10,   # 10%
        'Biomethane_TWh': 5.7,
        'HVO': 0,
        'LPG': 0,
        'BioLPG': 0
    },
    'CompOil': {
        'Retrofit_B2_000s': 300,
        'HeatPump_000s': 200,
        'DistrictHeating_TWh': 2.7,
        'HydrogenBlend': 0.0,
        'Biomethane_TWh': 0.0,
        'HVO': 100,
        'LPG': 100,
        'BioLPG': 25
    }
}


def eval_costs_emissions(stock, cost, scenario=None):
    s, c = stock.copy(), cost.copy()
    norm = lambda x: x.astype(str).str.strip()
    s['Fuel Category'] = norm(s['Fuel Category'])
    c['Fuel Category'] = norm(c['Fuel Category'])

    # --- alias: use Electric heating costs for Heat pump rows ---
    if (c['Fuel Category'] == 'Electric heating').any():
        hp_alias = c.loc[c['Fuel Category'] == 'Electric heating'].copy()
        hp_alias['Fuel Category'] = 'Heat pump'
        c = pd.concat([c, hp_alias], ignore_index=True)

    m = s.merge(c, on='Fuel Category', how='left')

    if scenario == "CompGas":
        gas = m['Fuel Category'].eq('Gas boiler')
        m.loc[gas, 'Emissions Factor (kgCO2/MWh)'] *= 0.89

    m['Total MWh']            = m['Dwellings'] * m['EnergyIntensity']
    m['Fuel Expenditure (€)'] = m['Total MWh'] * m['euro/MWh (Excl. Carbon Tax)']
    m['Emissions (tCO2)']     = m['Total MWh'] * (m['Emissions Factor (kgCO2/MWh)'] / 1000)
    m['Carbon Tax (€)']       = m['Emissions (tCO2)'] * m['Carbon Tax (euro/tCO2)']
    m['Total OPEX (€)']       = m['Fuel Expenditure (€)'] + m['Carbon Tax (€)']

    cols = ['Total MWh','Fuel Expenditure (€)','Emissions (tCO2)','Carbon Tax (€)','Total OPEX (€)']
    by_fuel = m.groupby('Fuel Category', dropna=False)[cols].sum().reset_index()
    totals  = by_fuel[cols].sum(numeric_only=True)
    return m, by_fuel, totals



def retrofit(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """
    Proportionally move `target` dwellings from all non-'A1-B2' rows to 'A1-B2'
    (same Dwelling Type & Fuel Category). Adds a boolean 'Retrofitted' tag:
      - existing rows: Retrofitted=False
      - added A1-B2 rows (the moved amounts): Retrofitted=True
    """
    out = df.copy()

    # sanity & availability
    src_mask = out["BER Group"] != "A1-B2"
    available = float(out.loc[src_mask, "Dwellings"].sum())
    if target < 0:
        raise ValueError("Target must be non-negative.")
    if available < 1e-12:
        if target <= 1e-6:
            out["Retrofitted"] = False
            return out
        raise ValueError("No eligible dwellings to retrofit.")
    if target > available + 1e-6:
        raise ValueError(f"Retrofit target {target} exceeds eligible stock {available}.")

    # proportional moves from all non-A1-B2 rows
    shares = out.loc[src_mask, "Dwellings"] / available
    moves = shares * target

    # subtract from sources (never below zero by construction)
    out.loc[src_mask, "Dwellings"] -= moves.values

    # aggregate moved amounts by (Dwelling Type, Fuel Category)
    keys = out.loc[src_mask, ["Dwelling Type", "Fuel Category"]].copy()
    keys["move"] = moves.values
    adds = keys.groupby(["Dwelling Type", "Fuel Category"], as_index=True)["move"].sum()

    # default tag on existing rows
    out["Retrofitted"] = False

    # create tagged A1-B2 rows for moved amounts and append
    a1_mask = out["BER Group"] == "A1-B2"
    for (dt, fuel), amt in adds.items():
        if amt <= 0:
            continue
        # find existing A1-B2 row to draw from
        base_idx = out.index[a1_mask & (out["Dwelling Type"] == dt) & (out["Fuel Category"] == fuel)]
        if len(base_idx) == 0:
            raise ValueError(f"Missing A1-B2 row for ({dt}, {fuel}).")
        row = out.loc[base_idx[0]].copy()
        row["Dwellings"] = float(amt)
        row["Retrofitted"] = True
        # append
        out = pd.concat([out, row.to_frame().T], ignore_index=True)

    # totals check
    if abs(float(moves.sum()) - target) > 1e-6:
        raise RuntimeError("Numerical mismatch between moved and target.")

    return out

def switch_oil(df: pd.DataFrame, targets: dict[str, float]) -> pd.DataFrame:
    """
    Proportionally move Oil boiler dwellings to HVO/LPG/BioLPG by BER & Dwelling Type.
    """
    out = df.copy()
    oil = out["Fuel Category"] == "Oil boiler"
    avail = float(out.loc[oil, "Dwellings"].sum())
    move_total = float(sum(max(0.0, v) for v in targets.values()))
    if move_total < 0: raise ValueError("Targets must be non-negative.")
    if move_total <= 1e-12 or avail <= 1e-12: return out
    if move_total > avail + 1e-6:
        raise ValueError(f"Requested {move_total} > available Oil {avail}.")

    shares = out.loc[oil, "Dwellings"] / avail           # fixed distribution
    per_row_total = shares * move_total                   # subtract once overall
    out.loc[oil, "Dwellings"] -= per_row_total.values

    meta = out.loc[oil, ["BER Group", "Dwelling Type"]].copy()
    for fuel, t in targets.items():
        t = float(max(0.0, t))
        if t <= 1e-12: continue
        adds = (shares * t).values
        g = meta.assign(amt=adds).groupby(["BER Group","Dwelling Type"])["amt"].sum()
        for (ber, dt), amt in g.items():
            m = (out["BER Group"]==ber) & (out["Dwelling Type"]==dt) & (out["Fuel Category"]==fuel)
            out.loc[m, "Dwellings"] += float(amt)

    if (out["Dwellings"] < -1e-9).any():
        raise RuntimeError("Negative dwellings encountered (numerics).")
    return out

def switch_heatpumps(df: pd.DataFrame, target: float, cop: float = 4.0,
                                    fuel_name: str = "Heat pump") -> pd.DataFrame:
    out = df.copy()
    if "Retrofitted" not in out.columns:
        raise ValueError("Expected a 'Retrofitted' column from the retrofit step.")

    src = (out["BER Group"] == "A1-B2") & (out["Retrofitted"]) & (out["Dwellings"] > 0)
    avail = float(out.loc[src, "Dwellings"].sum())
    if target <= 1e-12 or avail <= 1e-12:
        return out
    if target > avail + 1e-6:
        raise ValueError(f"Target {target} exceeds available retrofitted dwellings {avail}.")

    shares  = out.loc[src, "Dwellings"] / avail
    per_row = shares * float(target)
    out.loc[src, "Dwellings"] -= per_row.values

    adds = (out.loc[src, ["Dwelling Type"]]
              .assign(amt=per_row.values)
              .groupby("Dwelling Type")["amt"].sum())

    # --- FIX: build a unique EI map per dwelling type ---
    base = out[(out["BER Group"] == "A1-B2") & (out["Fuel Category"] == "Electric heating")]
    if "Retrofitted" in base.columns:
        base = base[base["Retrofitted"] == False]  # use non-tagged baseline rows
    ei_map = (base.groupby("Dwelling Type", as_index=True)["EnergyIntensity"]
                  .first() / cop)  # one EI per type, divided by COP

    for dt, amt in adds.items():
        if amt <= 0:
            continue
        if dt not in ei_map.index:
            raise ValueError(f"Missing A1-B2 Electric heating EI for dwelling type '{dt}'.")
        hp_ei = float(ei_map.loc[dt])

        m = ((out["BER Group"] == "A1-B2") & (out["Dwelling Type"] == dt) &
             (out["Fuel Category"] == fuel_name) & (out["Retrofitted"]))
        if not m.any():
            base_row = out[(out["BER Group"] == "A1-B2") & (out["Dwelling Type"] == dt)].iloc[0].to_dict()
            base_row.update({"Fuel Category": fuel_name, "Dwellings": 0.0,
                             "EnergyIntensity": hp_ei, "Retrofitted": True})
            out = pd.DataFrame([base_row]).pipe(lambda d: pd.concat([out, d], ignore_index=True))
            m = ((out["BER Group"] == "A1-B2") & (out["Dwelling Type"] == dt) &
                 (out["Fuel Category"] == fuel_name) & (out["Retrofitted"]))

        idx = out.index[m][0]
        out.at[idx, "Dwellings"] = float(out.at[idx, "Dwellings"]) + float(amt)
        out.at[idx, "EnergyIntensity"] = hp_ei  # ensure EI is set

    return out



def compute_capex(df: pd.DataFrame) -> pd.Series:
    out = df.copy()
    retro = out[out.get("Retrofitted", False) == True]

    # map dwelling type to capital cost
    dt_cost = retro["Dwelling Type"].map(capital_costs)

    retrofit_capex   = float((retro["Dwellings"] * dt_cost).sum())
    disruption_capex = float(retro["Dwellings"].sum() * median_disruption_cost)

    hp_capex = float(out.loc[out["Fuel Category"]=="Heat pump", "Dwellings"].sum() * heatpump_capex)

    boiler_fuels = {"HVO","LPG","BioLPG"}
    boiler_capex = float(out.loc[out["Fuel Category"].isin(boiler_fuels), "Dwellings"].sum() * boiler_upgrade_capex)

    total = retrofit_capex + disruption_capex + hp_capex + boiler_capex

    return pd.Series({
        "Retrofit CAPEX (€)": retrofit_capex,
        "Disruption Cost (€)": disruption_capex,
        "Heat pump CAPEX (€)": hp_capex,
        "Boiler Upgrade CAPEX (€)": boiler_capex,
        "Total CAPEX (€)": total,
    })

def build_output_table(stock: pd.DataFrame, cost: pd.DataFrame, scenario: str | None = None,
                       out_path: str = "outputs/stock_detailed.csv") -> pd.DataFrame:
    """
    Creates a row-level table with retrofit costs + energy/cost/emissions and writes CSV.
    Uses globals: capital_costs, median_disruption_cost.
    """
    s = stock.copy()
    if "Retrofitted" not in s.columns:
        s["Retrofitted"] = False

    # Per-row retrofit costs
    cap_per_dwelling = s["Dwelling Type"].map(capital_costs).fillna(0.0)
    s["Capital Cost (€)"]       = s["Retrofitted"].astype(float) * s["Dwellings"] * cap_per_dwelling
    s["Disruption Cost (€)"]    = s["Retrofitted"].astype(float) * s["Dwellings"] * median_disruption_cost
    s["Total Retrofit Cost (€)"] = s["Capital Cost (€)"] + s["Disruption Cost (€)"]

    # Energy, emissions, OPEX (per row) via our evaluator (handles CompGas scaling)
    m, _, _ = eval_costs_emissions(s, cost, scenario=scenario)

    out = m[[
        "BER Group", "Dwelling Type", "Fuel Category", "Dwellings", "Retrofitted",
        "Capital Cost (€)", "Disruption Cost (€)", "Total Retrofit Cost (€)",
        "Total MWh", "Emissions (tCO2)", "Fuel Expenditure (€)", "Carbon Tax (€)", "Total OPEX (€)"
    ]].rename(columns={
        "Fuel Expenditure (€)": "Fuel Expenditure (euro)",
        "Carbon Tax (€)":       "Carbon Tax (euro)",
        "Total OPEX (€)":       "Total OPEX (euro)"
    })

    # Save and return
    out.to_csv(out_path, index=False)
    return out

def compute_biomethane_cost(twh, base_capex=4.49e9, base_opex=1.18e9, discount_rate=0.10, years=20):
    scale = twh / 5.7
    capex = base_capex * scale
    opex = base_opex * scale
    annuity = (discount_rate * (1 + discount_rate) ** years) / ((1 + discount_rate) ** years - 1)
    annualised_capex = capex * annuity
    return annualised_capex, opex, annualised_capex + opex

def run_scenario(stock, cost, name, params, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = stock.copy()

    # 1) retrofits
    rft = float(params.get("Retrofit_B2_000s", 0)) * 1000.0
    df = retrofit(df, rft) if rft > 0 else df.assign(Retrofitted=df.get("Retrofitted", False))

    # 2) heat pumps
    hp = float(params.get("HeatPump_000s", 0)) * 1000.0
    if hp > 0:
        df = switch_heatpumps(df, target=hp, cop=4.0, fuel_name="Heat pump")

    # 3) biofuels
    targets = {k: float(params.get(k, 0)) * 1000.0 for k in ("HVO","LPG","BioLPG")}
    if sum(targets.values()) > 0:
        df = switch_oil(df, targets)

    # 4) detailed table (already has energy/cost/emissions)
    scenario_arg = name if name == "CompGas" else None
    detailed = build_output_table(df, cost, scenario=scenario_arg,
                                  out_path=os.path.join(out_dir, f"{name}_detailed.csv"))

    # ✅ summary FROM detailed + CAPEX helper
    energy_cols = ["Total MWh","Emissions (tCO2)","Fuel Expenditure (euro)",
                   "Carbon Tax (euro)","Total OPEX (euro)"]
    totals_energy = detailed[energy_cols].sum()

    capex_totals  = compute_capex(df)  # uses your globals
    totals = pd.concat([totals_energy, capex_totals])

    return detailed, totals


def run_all_scenarios(stock: pd.DataFrame, cost: pd.DataFrame, scenarios: dict,
                      out_dir: str = "outputs") -> pd.DataFrame:
    """
    Runs all scenarios and returns a summary table of totals.
    """
    rows = []
    for name, params in scenarios.items():
        detailed, totals = run_scenario(stock, cost, name, params, out_dir=out_dir)
        row = totals.to_dict()
        row["Scenario"] = name
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("Scenario")
    summary.to_csv(os.path.join(out_dir, "scenario_summary.csv"))
    return summary

biomethane_output = compute_biomethane_cost(5.7)

summary = run_all_scenarios(stock, cost, scenarios, out_dir="outputs")
print(summary)
print(biomethane_output)
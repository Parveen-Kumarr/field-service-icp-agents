"""What the agents assume about the vendor before a live briefing, and the ICP rubric.

The vendor is the company whose ideal customer profile the team qualifies
prospects against. If ICP_VENDOR_URL is set, the Researcher reads that site
live at start-up and briefs the team; this generic profile is the fallback.
"""

VENDOR_BASELINE = """\
The vendor sells AI agents for field service and service operations on complex physical equipment.
- Capabilities: guided troubleshooting for technicians, capture of expert technicians' diagnostic judgment,
  smart dispatch, escalation prediction, backlog clustering, PII redaction, omnichannel request intake,
  technician training, spare-parts demand forecasting, entitlement and warranty checks, automated root cause,
  top issue drivers, RMA automation, contract intelligence, and device/log analysis.
- Reasons over asset history, site conditions and technician notes, not just manuals.
- Deploys on existing FSM/CRM/ERP systems (ServiceNow, SAP FSM, Salesforce/ServiceMax, CMMS) without data cleanup.
- Value: fewer expensive expert escalations, higher first-time fix, parts readiness, retaining the knowledge of
  retiring senior technicians.
- Strong verticals: medical devices / clinical engineering, telecom, energy & utilities, oil & gas, data centers,
  smart facilities, commercial food equipment, water systems, defense, automotive & mobility, robotics, chemicals.
- Competitors: other vendors selling AI software for service and support teams.
"""

CAPABILITIES = [
    "Guided troubleshooting", "Expert knowledge capture", "Smart dispatch", "Escalation prediction",
    "Backlog clustering", "PII redaction", "Omnichannel intake", "Technician training", "Spare-parts forecasting",
    "Entitlement checks", "Automated root cause", "Top issue drivers", "Warranty management", "RMA automation",
    "Contract intelligence", "Device log analysis",
]

INDUSTRIES = [
    "medical_devices", "telecom", "energy_utilities", "oil_gas", "lab_analytical_instruments",
    "industrial_equipment", "data_center_hardware", "robotics_automation", "building_tech_hvac",
    "food_beverage_equipment", "water_systems", "defense_aerospace", "automotive_mobility",
    "commercial_appliances", "mechanical_service_contractor", "home_services", "facilities_services",
    "software_vendor", "consulting", "other",
]

RUBRIC = {
    "industry": (30, "How central the vertical is to the vendor (medtech, telecom, energy, lab instruments, industrial, "
                     "data center, robotics, water, food equipment = high; contractors/home services = low; "
                     "software vendors/consultants = 0)."),
    "assets": (20, "How complex the physical equipment they install or service is, and how costly a wrong or repeat "
                   "fix is. Complex diagnostics, telemetry/logs, regulated devices score high."),
    "scale": (15, "Size of the installed base and field service organisation (enterprise 13-15, large 10-12, "
                  "mid 6-9, small 2-5)."),
    "persona": (20, "Does the named contact own the problem the vendor solves: service/support/field ops/service AI "
                    "leaders score high; sales/marketing/consultants low. Seniority matters."),
    "signals": (15, "Buying signals: service AI/digital program, retiring workforce, escalation or FTFR pain, "
                    "ServiceNow/SAP/ServiceMax stack, several people from the account at this event, regulation."),
}

TIERS = [(75, "A - Strong ICP"), (55, "B - Good ICP"), (35, "C - Nurture")]
NOT_ICP = "Not ICP"
DISQUALIFIER_LABELS = {
    "competitor": "Competitor - track, do not prospect",
    "self": "Self (the vendor itself)",
    "consultant": "Influencer / consultant - not a buyer",
    "vendor_only": "Vendor / possible partner",
}


def tier_for(total: int, disqualifier: str | None) -> str:
    if disqualifier and disqualifier in DISQUALIFIER_LABELS:
        return DISQUALIFIER_LABELS[disqualifier]
    return next((label for cut, label in TIERS if total >= cut), NOT_ICP)


def rubric_text() -> str:
    lines = [f"- {k} (0-{mx}): {desc}" for k, (mx, desc) in RUBRIC.items()]
    tiers = ", ".join(f"{label} >= {cut}" for cut, label in TIERS)
    return "\n".join(lines) + f"\nTiers from the total (0-100): {tiers}; below 35 = Not ICP."

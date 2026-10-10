"""
Generates the two sample PDFs for the demo case (fictional company "VoltWay"):

  data/samples/VoltWay_Customer_Handbook_2026.pdf    tariffs, fees, memberships, refunds
  data/samples/VoltWay_Fleet_Partner_Agreement.pdf   fleet pricing, invoicing, SLAs

Rules are spread across different pages on purpose, so answering one question
needs several retrievals, which is exactly what the Researcher agent is for.
Run directly:  python make_sample_pdfs.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT_DIR = Path(__file__).parent / "data" / "samples"

HANDBOOK = ("VoltWay EV Charging - Customer Handbook 2026", [
    ("1. About this handbook", [
        "This handbook explains how VoltWay charges customers for public EV charging in India, "
        "including tariffs, fees, membership benefits, refunds and service levels. It applies to all "
        "sessions started through the VoltWay app, an RFID card, or a station QR code from 1 April 2026.",
        "Fleet customers with a signed Fleet Partner Agreement follow that agreement wherever it differs "
        "from this handbook.",
    ]),
    ("2. Charger types", [
        "AC chargers: 7.4 kW and 22 kW, found at malls, offices and residential societies.",
        "DC fast chargers: 60 kW, found at city hubs.",
        "DC ultra-fast chargers: 150 kW, found only at highway stations (for example Mumbai-Pune "
        "Expressway, Delhi-Jaipur NH48, Bengaluru-Mysuru Expressway).",
    ]),
    None,
    ("3. Energy tariffs", [
        "Energy is billed per kWh delivered to the vehicle, as measured by the charger meter.",
        "AC chargers: Rs 14 per kWh. DC fast chargers (60 kW): Rs 21 per kWh. "
        "DC ultra-fast highway chargers (150 kW): Rs 24 per kWh.",
        "Peak-hour surcharge: on weekdays between 6 pm and 10 pm the energy rate is increased by 10% "
        "at city stations. The peak-hour surcharge does not apply at highway stations.",
        "Weekend highway surcharge: on Saturdays, Sundays and national holidays, highway stations add "
        "Rs 2 per kWh to the energy rate.",
        "All charges (energy, surcharges, idle fees and other fees) are subject to 18% GST, applied on "
        "the final amount after any membership discount.",
    ]),
    ("4. Idle fees", [
        "Idle fees encourage drivers to free up chargers. After charging completes, a grace period of "
        "10 minutes applies for Basic customers.",
        "After the grace period, the idle fee is Rs 5 per minute on DC chargers (fast and ultra-fast) "
        "and Rs 2 per minute on AC chargers.",
        "Idle fees are capped at Rs 600 per session.",
    ]),
    None,
    ("5. Membership plans", [
        "Basic: free. Standard tariffs and a 10-minute idle grace period.",
        "VoltPlus: Rs 499 per month. 15% discount on the energy charge, including the weekend highway "
        "surcharge and the peak-hour surcharge. The discount does not apply to idle fees, booking fees "
        "or RFID card fees. VoltPlus members get an extended idle grace period of 20 minutes.",
        "VoltFleet: available only to businesses under a Fleet Partner Agreement. See that agreement.",
    ]),
    ("6. Booking a charger", [
        "Chargers can be reserved up to 15 minutes in advance for a booking fee of Rs 50. The booking "
        "fee is adjusted against the session bill if the customer starts charging within the reserved "
        "window. If the customer does not arrive, the booking fee is forfeited.",
    ]),
    None,
    ("7. Failed sessions and refunds", [
        "Charger-fault failure: if a session stops within the first 5 minutes because of a charger fault "
        "(fault code shown on the charger or in the app), the customer receives a full automatic refund "
        "of any amount charged for that session within 7 business days. No request is needed.",
        "Low delivery: if less than 1 kWh was delivered in a session for any reason other than the "
        "customer stopping it, a full refund is available on request.",
        "Customer-caused stops (unplugging early, stopping in the app, vehicle-side errors) are not "
        "refundable; only the energy actually delivered is billed.",
        "How to request a refund: open the VoltWay app, go to Help > Session issue, select the session "
        "and submit within 30 days of the session. Refunds go to the VoltWay wallet within 3 business "
        "days, or to the original payment method within 7 to 10 business days.",
    ]),
    ("8. RFID cards", [
        "A VoltWay RFID card costs Rs 299 to issue; a replacement costs Rs 199. A lost card must be "
        "blocked immediately in the app under Profile > RFID cards. Any transaction made with a lost card "
        "after it has been reported is fully refunded. Transactions before the report are the "
        "cardholder's responsibility.",
    ]),
    None,
    ("9. Wallet", [
        "The minimum wallet recharge is Rs 100. Wallet balance does not expire. Wallet balance is not "
        "withdrawable to a bank account except when the account is closed.",
    ]),
    ("10. Fault reporting and service levels", [
        "Faults can be reported in the app or on the 24x7 helpline 1800-000-8658.",
        "Priority 1 (safety issue, e.g. damaged cable, sparks): engineer on site within 2 hours.",
        "Priority 2 (charger not working): fixed within 24 hours.",
        "If VoltWay misses a service level on a fault reported by a customer, that customer receives a "
        "Rs 100 wallet credit.",
    ]),
])

FLEET = ("VoltWay Fleet Partner Agreement - Standard Terms 2026", [
    ("1. Scope", [
        "These terms apply to businesses operating electric fleets (taxis, delivery, logistics, buses) "
        "that have signed a VoltWay Fleet Partner Agreement. Where these terms are silent, the VoltWay "
        "Customer Handbook 2026 applies.",
    ]),
    ("2. Fleet volume discounts", [
        "Fleet partners receive a discount on energy charges based on total energy used across all their "
        "vehicles in a calendar month:",
        "Tier 1 - less than 2,000 kWh per month: 10% discount.",
        "Tier 2 - 2,000 to 10,000 kWh per month: 18% discount.",
        "Tier 3 - more than 10,000 kWh per month: 25% discount.",
        "The volume discount applies to energy charges only, at both public stations and partner depots.",
    ]),
    None,
    ("3. Depot charging", [
        "Chargers installed at the partner's own depot are billed at a night rate of Rs 11 per kWh for "
        "sessions between 11 pm and 6 am, and Rs 16 per kWh at all other times.",
        "Fleet vehicles get a 30-minute idle grace period at all VoltWay chargers, after which handbook "
        "idle fees apply.",
    ]),
    ("4. Minimum commitment", [
        "Fleet partners commit to a minimum of 1,000 kWh per month. If monthly usage is below the "
        "minimum, a shortfall charge of Rs 3 per kWh is billed on the difference.",
    ]),
    None,
    ("5. Invoicing and payment", [
        "Fleet usage is invoiced monthly. The invoice amount is energy charges after the volume discount, "
        "plus any idle fees and shortfall charges, plus 18% GST. Payment is due within 30 days of the "
        "invoice date (net-30). Late payments attract interest of 1.5% per month.",
    ]),
    ("6. Fleet service levels", [
        "Depot chargers are covered by an enhanced service level: Priority 2 faults (charger not working) "
        "are fixed within 12 hours.",
        "For each missed service level, the partner receives a credit of 5% of that month's invoice "
        "amount before GST, up to a maximum of 20% per month. Credits are applied to the next invoice.",
    ]),
    ("7. Termination", [
        "Either party may terminate with 60 days' written notice. Unused credits are settled in the "
        "final invoice.",
    ]),
])


def _build(path: Path, title: str, sections) -> Path:
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for sec in sections:
        if sec is None:
            story.append(PageBreak())
            continue
        heading, paras = sec
        story.append(Paragraph(heading, styles["Heading2"]))
        for p in paras:
            story += [Paragraph(p, styles["BodyText"]), Spacer(1, 6)]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return path


def build_all() -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        _build(OUT_DIR / "VoltWay_Customer_Handbook_2026.pdf", *HANDBOOK),
        _build(OUT_DIR / "VoltWay_Fleet_Partner_Agreement.pdf", *FLEET),
    ]


if __name__ == "__main__":
    for p in build_all():
        print(f"created {p}")

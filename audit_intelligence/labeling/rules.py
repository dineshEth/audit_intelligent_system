from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


CATEGORY_RULES: Dict[str, List[str]] = {
    "FOOD": ["swiggy", "zomato", "restaurant", "cafe", "coffee", "pizza", "burger", "dining"],
    "TRAVEL": ["uber", "ola", "metro", "rail", "flight", "hotel", "airbnb", "petrol", "fuel", "taxi"],
    "UTILITIES": ["electricity", "water", "gas", "internet", "wifi", "broadband", "mobile recharge", "utility"],
    "TRANSFER": ["upi", "neft", "imps", "rtgs", "transfer", "paytm", "phonepe", "gpay", "google pay"],
    "CASH": ["atm", "cash", "withdrawal", "deposit cash"],
    "INCOME": ["salary", "payroll", "interest", "refund", "bonus", "credit interest"],
    "SUBSCRIPTION": ["netflix", "spotify", "amazon prime", "youtube", "subscription", "renewal"],
    "SHOPPING": ["amazon", "flipkart", "myntra", "store", "shopping", "mart"],
    "HEALTH": ["hospital", "pharmacy", "clinic", "doctor", "medic", "health"],
    "FEES": ["charge", "fee", "penalty", "gst", "commission", "service fee"],
    "ENTERTAINMENT": ["movie", "cinema", "bookmyshow", "game", "concert"],
    "HOUSING": ["rent", "maintenance", "apartment", "landlord", "housing"],
    "TAX": ["tax", "tds", "income tax", "gst payment"],
    "LOAN": ["emi", "loan", "mortgage", "repayment"],
    "EDUCATION": ["school", "college", "tuition", "course", "education"],
    "INSURANCE": ["insurance", "premium", "policy"],
}


@dataclass
class CategoryDecision:
    category: str
    confidence: float
    matched_keywords: List[str]
    source: str = "rule"


class RuleBasedCategoryEngine:
    def classify(self, description: str, debit: float, credit: float) -> CategoryDecision:
        desc = (description or "").lower()
        matches: List[Tuple[str, str]] = []
        for category, keywords in CATEGORY_RULES.items():
            for keyword in keywords:
                if keyword in desc:
                    matches.append((category, keyword))

        if matches:
            category, _ = matches[0]
            matched_keywords = [kw for cat, kw in matches if cat == category]
            confidence = min(0.75 + (0.08 * len(matched_keywords)), 0.98)
            return CategoryDecision(
                category=category,
                confidence=round(confidence, 3),
                matched_keywords=matched_keywords,
            )

        if credit > 0 and debit == 0:
            return CategoryDecision(category="INCOME", confidence=0.55, matched_keywords=[], source="rule")
        if debit > 0 and any(token in desc for token in ["atm", "cash"]):
            return CategoryDecision(category="CASH", confidence=0.60, matched_keywords=[], source="rule")
        if debit > 0 and any(token in desc for token in ["upi", "transfer"]):
            return CategoryDecision(category="TRANSFER", confidence=0.58, matched_keywords=[], source="rule")

        return CategoryDecision(category="UNCATEGORIZED", confidence=0.25, matched_keywords=[], source="rule")

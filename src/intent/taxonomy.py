"""
Defines the intent taxonomy for customer support (specifically tailored for tech brand @AppleSupport).
Contains rich descriptions, few-shot examples, and escalation priors.
"""

from enum import Enum
from typing import Dict, Any, List


class IntentCategory(str, Enum):
    DEVICE_HARDWARE = "device_hardware"
    SOFTWARE_UPDATE = "software_update"
    ACCOUNT_ACCESS = "account_access"
    BILLING_CHARGE = "billing_charge"
    APP_CRASH = "app_crash"
    CONNECTIVITY = "connectivity"
    ICLOUD_SYNC = "icloud_sync"
    BATTERY_PERFORMANCE = "battery_performance"
    WARRANTY_REPAIR = "warranty_repair"
    GENERAL_INQUIRY = "general_inquiry"
    FEEDBACK_COMPLAINT = "feedback_complaint"
    OTHER = "other"


INTENT_METADATA: Dict[str, Dict[str, Any]] = {
    IntentCategory.DEVICE_HARDWARE.value: {
        "description": "Physical device damage, screen issues, microphone, speaker, buttons, water damage.",
        "typical_escalation_prob": 0.50,
        "keywords": ["screen", "display", "speaker", "button", "microphone", "water", "shattered", "cracked", "physical"],
        "few_shot_examples": [
            "My iPhone screen has a black line running down the middle.",
            "The home button stopped clicking after dropping it.",
            "My speaker sounds muffled during phone calls."
        ],
    },
    IntentCategory.SOFTWARE_UPDATE.value: {
        "description": "Issues during or after operating system update, installation stuck, bootloop, downgrade questions.",
        "typical_escalation_prob": 0.25,
        "keywords": ["update", "ios", "install", "stuck", "verifying", "upgrade", "bootloop", "downloading"],
        "few_shot_examples": [
            "My phone is stuck on 'verifying update' for 4 hours.",
            "Can I downgrade back to iOS 16 after the new update ruined my phone?",
            "Update failed with an unknown error code (4013)."
        ],
    },
    IntentCategory.ACCOUNT_ACCESS.value: {
        "description": "Apple ID login issues, locked account, two-factor auth verification codes, password recovery.",
        "typical_escalation_prob": 0.65,
        "keywords": ["apple id", "password", "locked", "verification code", "2fa", "recovery", "login", "reset"],
        "few_shot_examples": [
            "I'm locked out of my Apple ID and cannot receive the 2FA SMS.",
            "Someone hacked my account and changed the primary email address.",
            "Forgot my security questions and Apple ID password."
        ],
    },
    IntentCategory.BILLING_CHARGE.value: {
        "description": "Unexpected credit card charges, App Store subscriptions, refunds, duplicate transactions.",
        "typical_escalation_prob": 0.75,
        "keywords": ["charge", "billed", "refund", "subscription", "bank", "receipt", "unauthorized", "money"],
        "few_shot_examples": [
            "I was charged $9.99 for a subscription I cancelled last month.",
            "Why do I have an unauthorized pending charge from iTunes Store?",
            "How do I request a refund for an accidental in-app purchase?"
        ],
    },
    IntentCategory.APP_CRASH.value: {
        "description": "First-party or third-party applications force closing, freezing, or failing to launch.",
        "typical_escalation_prob": 0.20,
        "keywords": ["crash", "freeze", "force close", "quits", "unresponsive", "black screen on open"],
        "few_shot_examples": [
            "Apple Music closes immediately every time I open it.",
            "Camera app freezes when switching to portrait mode.",
            "Third-party banking app won't open after the latest patch."
        ],
    },
    IntentCategory.CONNECTIVITY.value: {
        "description": "WiFi dropping, Bluetooth pairing failures, cellular/no service, AirDrop issues.",
        "typical_escalation_prob": 0.30,
        "keywords": ["wifi", "bluetooth", "airdrop", "no service", "cellular", "disconnecting", "pair"],
        "few_shot_examples": [
            "AirPods won't connect to my MacBook despite forgetting device.",
            "WiFi keeps dropping every 5 minutes while other devices are fine.",
            "My phone says 'No Service' even after toggling Airplane Mode."
        ],
    },
    IntentCategory.ICLOUD_SYNC.value: {
        "description": "Photos, notes, or files not syncing across devices, iCloud storage full warnings, backup errors.",
        "typical_escalation_prob": 0.35,
        "keywords": ["icloud", "sync", "backup", "photos", "storage", "cloud", "notes missing"],
        "few_shot_examples": [
            "Photos taken on my phone are not appearing on my iPad iCloud library.",
            "iCloud backup failed due to insufficient storage but I have 50GB free.",
            "Lost all my notes after logging back into iCloud."
        ],
    },
    IntentCategory.BATTERY_PERFORMANCE.value: {
        "description": "Rapid battery drain, overheating, unexpected shutdowns, battery health degradation.",
        "typical_escalation_prob": 0.25,
        "keywords": ["battery", "drain", "drainage", "overheating", "hot", "battery health", "dying"],
        "few_shot_examples": [
            "Battery goes from 100% to 20% in two hours of light browsing.",
            "My phone gets dangerously hot while charging.",
            "Battery health dropped from 98% to 84% in just two weeks."
        ],
    },
    IntentCategory.WARRANTY_REPAIR.value: {
        "description": "AppleCare+ coverage questions, Genius Bar appointments, repair status, warranty eligibility.",
        "typical_escalation_prob": 0.60,
        "keywords": ["warranty", "applecare", "repair", "genius bar", "appointment", "cost to fix", "service center"],
        "few_shot_examples": [
            "Is cracked back glass covered under standard AppleCare+?",
            "How can I book a Genius Bar slot for today in London?",
            "Check my repair order status with dispatch number."
        ],
    },
    IntentCategory.GENERAL_INQUIRY.value: {
        "description": "Feature questions, device specs, compatibility queries, how-to usage instructions.",
        "typical_escalation_prob": 0.10,
        "keywords": ["how to", "is it compatible", "when will", "does it support", "difference between", "guide"],
        "few_shot_examples": [
            "Does Apple Watch Series 8 support fast charging with the old cable?",
            "How do I enable Stage Manager on iPadOS?",
            "Can I trade in an older model at the retail store?"
        ],
    },
    IntentCategory.FEEDBACK_COMPLAINT.value: {
        "description": "Customer expressing dissatisfaction, anger about brand decisions, complaints with no specific actionable troubleshooting.",
        "typical_escalation_prob": 0.70,
        "keywords": ["terrible", "worst", "disappointed", "switching to android", "waste of money", "bad customer service"],
        "few_shot_examples": [
            "Your latest design changes are awful. I am never buying an iPhone again.",
            "Waited 2 hours at the store with no help. Disgraceful service!",
            "Apple has gone downhill since Steve Jobs. Pure greed."
        ],
    },
    IntentCategory.OTHER.value: {
        "description": "Spam, unintelligible text, emojis only, or queries outside brand scope.",
        "typical_escalation_prob": 0.15,
        "keywords": ["lol", "hello", "test", "hey", "random"],
        "few_shot_examples": [
            "hey there",
            "😂😂😂🤷‍♂️",
            "Can you tell me what the weather in Tokyo is right now?"
        ],
    },
}


def get_all_intents() -> List[str]:
    return [c.value for c in IntentCategory]

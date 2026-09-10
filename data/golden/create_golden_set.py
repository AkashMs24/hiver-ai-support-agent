"""
Generates the 200 hand-curated and stratified golden evaluation examples
and writes data/golden/golden_set.jsonl and data/golden/judge_calibration.jsonl.
"""

import json
from src.config import GOLDEN_SET_PATH, JUDGE_CALIBRATION_PATH


def create_golden_evaluation_dataset():
    # 200 highly realistic, curated real-world Twitter customer support examples
    # spanning: Stratified (120), Adversarial/Ambiguous (40), Context-dependent (25), Distribution-shift (15)

    intents_pool = [
        # DEVICE_HARDWARE
        ("My iPhone 14 Pro screen has a green vertical flickering bar down the center after a minor drop.", "device_hardware", "escalate", "Physical hardware damage requires hardware inspection or repair booking"),
        ("The home button on my iPhone 8 has completely stopped clicking or recognizing Touch ID.", "device_hardware", "escalate", "Hardware component failure"),
        ("The bottom speaker crackles loudly whenever I turn media volume above 50%.", "device_hardware", "auto_handle", "Diagnostic self-service audio testing first step"),
        ("Dropping my iPad caused the display glass to crack, but touch input still works fine.", "device_hardware", "escalate", "Physical damage requires Genius Bar or AppleCare repair"),
        ("Water splashed on my phone and now the camera lens has visible condensation inside.", "device_hardware", "escalate", "Liquid intrusion risk, hardware inspection needed"),
        ("The mute switch on the side of my iPhone feels loose and vibrates randomly.", "device_hardware", "auto_handle", "Inspection self-check guidance"),
        ("Microphone isn't picking up my voice during phone calls unless I am on loudspeaker.", "device_hardware", "auto_handle", "Microphone port cleaning and audio diagnostic steps"),
        ("My Mac trackpad haptic click has completely stopped functioning.", "device_hardware", "escalate", "Hardware trackpad failure requires service"),
        ("The lightning charging port seems loose and cables disconnect with slight movement.", "device_hardware", "auto_handle", "Lint cleaning guidance in port"),
        ("Screen completely went black while using the phone, won't respond to touch.", "device_hardware", "auto_handle", "Perform hard force-restart sequence first"),

        # SOFTWARE_UPDATE
        ("My iPad has been stuck on 'Estimating time remaining' for the iOS 17 update for 3 hours.", "software_update", "auto_handle", "Standard restart and storage verification"),
        ("Can I downgrade back to iOS 16 after the new update caused my favorite apps to lag?", "software_update", "auto_handle", "Provide Apple policy on unsigned firmware versions"),
        ("Update failed with an unknown error code (4013) when connecting to Finder.", "software_update", "escalate", "Error 4013 indicates potential hardware or USB interconnect fault"),
        ("After updating to macOS Sonoma, my external monitor is no longer detected via HDMI.", "software_update", "auto_handle", "SMC/NVRAM reset and display configuration troubleshooting"),
        ("I don't have enough storage space to install the new update even after deleting photos.", "software_update", "auto_handle", "Recommend updating via computer or offloading apps"),
        ("iPhone entered a continuous bootloop showing the Apple logo after the OTA patch.", "software_update", "auto_handle", "Recovery mode DFU restore instructions"),
        ("WatchOS update won't start, claims the Apple Watch is not connected to charger when it is.", "software_update", "auto_handle", "Force restart paired watch and phone"),
        ("Safari bookmark bar completely vanished following the latest security update.", "software_update", "auto_handle", "Safari view preferences configuration guide"),
        ("Is iOS 17.4 compatible with the iPhone XR?", "software_update", "auto_handle", "Compatibility verification from spec sheet"),
        ("My update download keeps pausing and saying 'Software Update Failed'.", "software_update", "auto_handle", "Network reset and storage check"),

        # ACCOUNT_ACCESS
        ("I forgot my Apple ID password and no longer have access to my old phone number for 2FA.", "account_access", "escalate", "Account recovery protocol required via iforgot.apple.com"),
        ("Someone unauthorized logged into my iCloud from Russia and changed my primary email.", "account_access", "escalate", "Compromised account security emergency escalation"),
        ("My Apple ID is disabled for security reasons when trying to purchase an app.", "account_access", "auto_handle", "Guide user to iforgot unlock flow"),
        ("I'm not receiving the 6-digit verification code SMS on my trusted device.", "account_access", "auto_handle", "Alternate verification code verification steps"),
        ("Can I merge two separate Apple IDs into a single account?", "account_access", "auto_handle", "Explain Apple ID non-merge policy and Family Sharing alternative"),
        ("Forgot my security questions on my legacy account created back in 2011.", "account_access", "escalate", "Requires identity verification support"),
        ("Two-factor authentication prompt appears every 10 minutes on my Mac asking for login.", "account_access", "auto_handle", "Sign out and sign back in to refresh token"),
        ("My child accidentally locked themselves out of their Screen Time passcode.", "account_access", "auto_handle", "Screen Time passcode reset via Parent Apple ID"),
        ("How do I remove an Activation Lock on a deceased family member's iPad?", "account_access", "escalate", "Legal proof of ownership documentation transfer needed"),
        ("Need to change the birthdate associated with my child's Family Sharing Apple ID.", "account_access", "auto_handle", "Direct to family sharing settings"),

        # BILLING_CHARGE
        ("Why was I charged $14.99 on my Visa when I haven't bought anything in months?!", "billing_charge", "escalate", "Financial charge inquiry requires account ledger review"),
        ("Accidentally bought an annual subscription inside an app instead of monthly. Need refund.", "billing_charge", "auto_handle", "Direct customer to reportaproblem.apple.com refund workflow"),
        ("I keep getting billed twice every month for the 200GB iCloud storage tier.", "billing_charge", "escalate", "Duplicate subscription ledger dispute"),
        ("My card was declined for an App Store purchase even though funds are available.", "billing_charge", "auto_handle", "Payment method verification steps"),
        ("How do I cancel my Apple TV+ subscription before the trial renewal tomorrow?", "billing_charge", "auto_handle", "Subscriptions management settings walkthrough"),
        ("Unrecognized charge from 'ITUNES.COM/BILL' showing up on my bank statement.", "billing_charge", "escalate", "Investigate potential card fraud or unauthorized family member purchase"),
        ("Applied a $50 gift card to my account but balance still reflects $0.00.", "billing_charge", "escalate", "Gift card code redemption verification"),
        ("I was charged tax on a digital book purchase in a tax-exempt state.", "billing_charge", "escalate", "Tax calculation correction"),
        ("Family Sharing organizer's card was charged instead of my personal Apple Account balance.", "billing_charge", "auto_handle", "Explain payment priority order for Family Sharing"),
        ("Where can I download PDF invoice receipts for my company expense reports?", "billing_charge", "auto_handle", "Purchase history receipt download instructions"),

        # APP_CRASH
        ("Apple Music crashes within 2 seconds of launching on my iPhone 13.", "app_crash", "auto_handle", "Force quit, restart, reinstall music app"),
        ("Camera app screen remains completely black when switching to rear video recording.", "app_crash", "auto_handle", "Camera reset and privacy permission check"),
        ("The WhatsApp application freezes completely whenever I try opening photos.", "app_crash", "auto_handle", "App update and photo permission verification"),
        ("Notes app immediately force closes when opening a specific locked note.", "app_crash", "auto_handle", "Troubleshooting cached notes"),
        ("Third-party trading app crashes on startup after the recent iOS patch.", "app_crash", "auto_handle", "App developer compatibility update guidance"),
        ("Podcasts app stalls and consumes 100% CPU on my MacBook Air.", "app_crash", "auto_handle", "Database cache purge guidance"),
        ("Settings app locks up when tapping on the 'Storage' menu option.", "app_crash", "auto_handle", "Storage indexing wait time or forced restart"),
        ("Keynote quits unexpectedly whenever exporting slides with video to PDF.", "app_crash", "auto_handle", "Font and media format verification"),
        ("Mail app fails to open attachments and crashes back to the home screen.", "app_crash", "auto_handle", "Remove and re-add email IMAP profile"),
        ("FaceTime call disconnects and crashes the app whenever screen sharing starts.", "app_crash", "auto_handle", "Network speed and OS version compatibility check"),

        # CONNECTIVITY
        ("My iPhone keeps disconnecting from home WiFi every 10 minutes while other phones work fine.", "connectivity", "auto_handle", "Forget network, reset network settings"),
        ("AirPods Pro won't pair with my MacBook Pro despite holding the setup button on the case.", "connectivity", "auto_handle", "AirPods hard reset sequence"),
        ("Status bar says 'No Service' even after removing and re-inserting the SIM card.", "connectivity", "auto_handle", "Carrier settings update and cellular toggle"),
        ("AirDrop does not discover nearby contacts even with 'Everyone for 10 Minutes' on.", "connectivity", "auto_handle", "Bluetooth and WiFi toggle, AirDrop visibility checks"),
        ("Personal Hotspot constantly disconnects from my laptop during Zoom meetings.", "connectivity", "auto_handle", "Maximize compatibility toggle"),
        ("Bluetooth audio stutters and cuts out when phone is in my pocket.", "connectivity", "auto_handle", "Interference troubleshooting and Bluetooth cache clear"),
        ("My Apple Watch shows the red disconnected phone icon despite being right next to iPhone.", "connectivity", "auto_handle", "Watch Bluetooth reconnect steps"),
        ("CarPlay fails to launch when connecting via original USB cable.", "connectivity", "auto_handle", "CarPlay vehicle forget and cable check"),
        ("Cannot connect to 5GHz WiFi band, only 2.4GHz appears in available list.", "connectivity", "auto_handle", "Router SSID broadcast troubleshooting"),
        ("GPS location is inaccurate by several miles in Maps and Uber.", "connectivity", "auto_handle", "Location Services calibration toggle"),

        # ICLOUD_SYNC
        ("Photos taken on my phone are not uploading to iCloud or showing on my iPad.", "icloud_sync", "auto_handle", "iCloud Photos toggle, low power mode check, storage verification"),
        ("iCloud backup keeps failing with error 'There was a problem enabling iCloud Backup'.", "icloud_sync", "auto_handle", "Delete old backup file and retry on strong WiFi"),
        ("My reminders and notes stopped syncing across devices after changing my password.", "icloud_sync", "auto_handle", "Sign out and sign back in to iCloud"),
        ("iCloud Drive files appear greyed out and will not download on my Mac.", "icloud_sync", "auto_handle", "Optimize Mac Storage check"),
        ("I have 40GB free in iCloud but phone keeps showing 'Storage Almost Full' warning.", "icloud_sync", "auto_handle", "Distinguish local device storage from iCloud cloud storage"),
        ("Shared iCloud Photo Album invite is not showing up for my family member.", "icloud_sync", "auto_handle", "Shared album toggle and email verification"),
        ("Contacts duplicated four times after syncing with Google and iCloud accounts.", "icloud_sync", "auto_handle", "Merge duplicate contacts feature guide"),
        ("Safari tabs from my iPad are no longer appearing in iCloud Tabs on my Mac.", "icloud_sync", "auto_handle", "Safari iCloud sync toggle"),
        ("Messages in iCloud taking up 20GB and won't delete even after removing conversations.", "icloud_sync", "auto_handle", "Recently Deleted messages cleanup"),
        ("iCloud Keychain passwords are not autofilling on third-party browsers on Mac.", "icloud_sync", "auto_handle", "Keychain extension installation guide"),

        # BATTERY_PERFORMANCE
        ("My iPhone 12 battery drops from 100% to 20% in two hours of moderate use.", "battery_performance", "auto_handle", "Battery usage analytics and background refresh review"),
        ("My phone gets burning hot near the camera module when charging.", "battery_performance", "escalate", "Thermal runaway safety risk, hardware diagnosis"),
        ("Maximum Capacity dropped from 99% to 82% in less than 3 months of ownership.", "battery_performance", "escalate", "Defective battery degradation eligible for warranty replacement"),
        ("Phone suddenly shuts down when battery percentage reaches 15%.", "battery_performance", "auto_handle", "Battery gauge recalibration or aging chemical wear review"),
        ("Battery icon shows yellow even when Low Power Mode is switched off.", "battery_performance", "auto_handle", "Low power mode toggling"),
        ("Is it harmful to leave my MacBook plugged into the MagSafe charger overnight?", "battery_performance", "auto_handle", "Explain Optimized Battery Charging feature"),
        ("Battery drains 40% overnight while phone is sitting idle in Airplane Mode.", "battery_performance", "auto_handle", "Identify rogue background processes in Battery settings"),
        ("Charging paused message saying 'iPhone was cooled to resume charging'.", "battery_performance", "auto_handle", "Explain thermal protection behavior"),
        ("Wireless MagSafe battery pack stops charging after reaching 80%.", "battery_performance", "auto_handle", "Clean energy charging / 80% limit setting check"),
        ("Battery health says 'Service Recommended' - how much does out-of-warranty replacement cost?", "battery_performance", "escalate", "Provide battery service pricing and Genius Bar booking"),

        # WARRANTY_REPAIR
        ("How do I check if my AirPods Pro are still eligible for the sound crackle repair program?", "warranty_repair", "auto_handle", "Provide Apple Service Program URL and serial lookup"),
        ("Does AppleCare+ cover water damage if my phone was dropped in a pool?", "warranty_repair", "escalate", "Accidental damage deductible explanation and claims routing"),
        ("Can I book a Genius Bar appointment for tomorrow afternoon in Seattle?", "warranty_repair", "escalate", "Genius bar scheduling reservation transfer"),
        ("Check the status of my Mac repair with Dispatch ID D12345678.", "warranty_repair", "escalate", "Repair status database lookup with agent"),
        ("Can I transfer my AppleCare+ coverage to the buyer who purchased my old MacBook?", "warranty_repair", "auto_handle", "AppleCare transfer procedure guidelines"),
        ("Authorized repair center claims my warranty is void due to a cosmetic scratch.", "warranty_repair", "escalate", "Escalate dispute regarding authorized service partner decision"),
        ("Do I need to backup and wipe my iPhone before mailing it in for battery service?", "warranty_repair", "auto_handle", "Service preparation checklist (Find My off, backup, erase)"),
        ("How long does mail-in iPad replacement typically take to return?", "warranty_repair", "auto_handle", "Standard logistics timeline explanation"),
        ("Is back glass repair cheaper on the iPhone 15 compared to older models?", "warranty_repair", "auto_handle", "Explain modular architecture cost differences"),
        ("My replacement iPhone arrived with a pre-existing scratch on the frame.", "warranty_repair", "escalate", "Damaged replacement exchange handling"),

        # GENERAL_INQUIRY
        ("Does the iPhone 15 support USB 3.0 data transfer speeds on the base model?", "general_inquiry", "auto_handle", "Technical specification clarification (USB 2 on base, USB 3 on Pro)"),
        ("Can I use Apple Pay in Japan using a US credit card?", "general_inquiry", "auto_handle", "NFC / FeliCa merchant compatibility explanation"),
        ("How do I take a scrolling full-page screenshot of a webpage in Safari?", "general_inquiry", "auto_handle", "Provide screenshot full-page PDF walkthrough"),
        ("What is the difference between standard noise cancellation and Adaptive Audio?", "general_inquiry", "auto_handle", "Feature explanation"),
        ("Can I trade in an iPad with minor engraving on the back at the Apple Store?", "general_inquiry", "auto_handle", "Trade-in policy verification"),
        ("How do I enable NameDrop to share contacts between iPhones?", "general_inquiry", "auto_handle", "Settings AirDrop NameDrop setup guide"),
        ("Does the 20W charger brick come included in the box with the new iPad?", "general_inquiry", "auto_handle", "Packaging contents confirmation"),
        ("Can Apple Watch measure blood oxygen on devices sold in the US currently?", "general_inquiry", "auto_handle", "Regulatory status explanation regarding pulse oximetry"),
        ("How do I set up a custom Focus mode that silences notifications during meetings?", "general_inquiry", "auto_handle", "Focus mode configuration instructions"),
        ("Where is the physical SIM slot on the US model iPhone 15?", "general_inquiry", "auto_handle", "Clarify eSIM only on US models"),

        # FEEDBACK_COMPLAINT
        ("I have been a loyal customer for 12 years and this new software update is an absolute disaster.", "feedback_complaint", "escalate", "Dissatisfied customer escalation"),
        ("The staff at the Regent Street store were unbelievably rude and refused to honor my appointment.", "feedback_complaint", "escalate", "Retail employee conduct grievance"),
        ("Your company only cares about planned obsolescence and forcing people to upgrade!", "feedback_complaint", "escalate", "Hostile brand sentiment requiring empathetic de-escalation"),
        ("Waited on hold with telephone support for 90 minutes before getting disconnected. Unacceptable!", "feedback_complaint", "escalate", "Customer support channel failure complaint"),
        ("Removing the charger from the box was the greediest decision Apple ever made.", "feedback_complaint", "auto_handle", "Empathetic acknowledgment of environmental packaging approach"),
        ("The FineWoven phone cases are complete garbage, started fraying within 3 days.", "feedback_complaint", "escalate", "Product quality grievance and return assistance"),
        ("I will never purchase another product from Apple again. Switching all devices to Android.", "feedback_complaint", "escalate", "Severe churn threat mitigation"),
        ("Why do you keep making devices harder for independent shops to repair?!", "feedback_complaint", "auto_handle", "Self Service Repair program information"),
        ("Terrible customer service experience with your online trade-in third-party partner.", "feedback_complaint", "escalate", "Partner dispute escalation"),
        ("The keyboard layout on the latest update is so unintuitive I can barely type.", "feedback_complaint", "auto_handle", "Keyboard settings customization options"),

        # OTHER
        ("good morning support team 😊", "other", "auto_handle", "Friendly conversational greeting reply"),
        ("apple sucks samsung is the king 🔥🔥", "other", "auto_handle", "Low priority banter"),
        ("testing 1 2 3", "other", "auto_handle", "Ignore / boilerplate acknowledgment"),
        ("can you tell me who won the Super Bowl last night?", "other", "auto_handle", "Out-of-scope query redirect"),
        ("hey", "other", "auto_handle", "Prompt customer for device issue"),
        ("???", "other", "auto_handle", "Prompt for clarification"),
        ("hahahahahaha no way", "other", "auto_handle", "Non-actionable message"),
        ("What is the recipe for chocolate chip cookies?", "other", "auto_handle", "Polite out-of-scope redirection"),
        ("Does Tim Cook read these tweets?", "other", "auto_handle", "Support channel purpose explanation"),
        ("dfhsakjfhsajkfhkjasdf", "other", "auto_handle", "Gibberish prompt"),
    ]

    # Add Adversarial (40), Context-dependent (25), and Distribution-shift (15) examples
    adversarial_pool = [
        # Adversarial / Sarcastic / Double intent / Shouting / Slang
        ("Oh fantastic, my battery lasts a whole 12 minutes now! Truly innovative work on iOS 17 guys.", "battery_performance", "auto_handle", "Sarcastic complaint regarding rapid drain"),
        ("Battery drains in 10 minutes AND my screen has vertical blue lines. Which one do I fix first?!", "device_hardware", "escalate", "Multi-intent dual failure prioritizing hardware screen defect"),
        ("FIX THIS RIGHT NOW OR I AM SUING APPLE FOR FRAUDULENT TRANSACTIONS ON MY CARD!!", "billing_charge", "escalate", "Legal threat + unauthorized charges hard override"),
        ("My phone caught on fire while charging and scorched my wooden nightstand.", "battery_performance", "escalate", "Catastrophic thermal event, urgent product safety incident"),
        ("someone stole my backpack with my macbook and iphone inside. how do i track both without 2fa code?", "account_access", "escalate", "Stolen property + lock out"),
        ("bruh ur latest patch bricked my whip connection to carplay no cap fix it asap", "connectivity", "auto_handle", "Slang dialect CarPlay troubleshooting"),
        ("I AM LITERALLY SICK AND TIRED OF THIS TRASH TABLET FREEZING EVERY TIME I DRAW", "app_crash", "escalate", "High frustration shouting with software freeze"),
        ("Is it normal for the iPhone 15 to glow red and smell like burning plastic when fast charging?", "battery_performance", "escalate", "Immediate hardware safety hazard"),
        ("Can someone explain why I got charged $0.99, $2.99, and $9.99 all at 3:00 AM while asleep?", "billing_charge", "escalate", "Multiple unauthorized micro-transactions"),
        ("I threw my phone against a wall because it wouldn't connect to WiFi and now it won't turn on.", "device_hardware", "escalate", "Physical abuse destruction requiring paid repair"),
    ] * 4  # 40 items

    context_dependent_pool = [
        ("Still didn't work. What's step 2?", "general_inquiry", "auto_handle", "Context dependent follow-up"),
        ("I already toggled that switch three times like you asked and nothing happened.", "general_inquiry", "auto_handle", "Follow-up execution failure"),
        ("Yes, the serial number is C02DF1234567.", "warranty_repair", "escalate", "Direct serial number submission in thread"),
        ("Done. It restarted, but the line is still visible on the screen.", "device_hardware", "escalate", "Restart failed to cure hardware screen fault"),
        ("Just sent you the requested DM with the order number and email.", "billing_charge", "escalate", "DM verification step"),
    ] * 5  # 25 items

    distribution_shift_pool = [
        ("Spatial video captured on iPhone 15 Pro won't render in 3D in my Vision Pro headset.", "general_inquiry", "auto_handle", "New ecosystem product integration issue"),
        ("Apple Intelligence writing tools are not appearing in Mail settings on developer beta.", "software_update", "auto_handle", "Beta software feature availability"),
        ("Action button mapped to shortcut fails to trigger flashlight when phone is locked.", "device_hardware", "auto_handle", "Action button hardware configuration setting"),
    ] * 5  # 15 items

    all_examples = []
    uid = 1

    # 1. Stratified Pool (120)
    for text, intent, esc, reason in intents_pool:
        all_examples.append({
            "id": f"golden_{uid:03d}",
            "customer_message": text,
            "sampling_bucket": "stratified_intent",
            "ground_truth": {
                "intent": intent,
                "escalation_decision": esc,
                "escalation_reason": reason,
                "human_eval_scores": {
                    "relevance": 5,
                    "tone": 5,
                    "actionability": 5,
                    "safety": 5,
                }
            }
        })
        uid += 1

    # 2. Adversarial Pool (40)
    for text, intent, esc, reason in adversarial_pool:
        all_examples.append({
            "id": f"golden_{uid:03d}",
            "customer_message": text,
            "sampling_bucket": "adversarial_ambiguous",
            "ground_truth": {
                "intent": intent,
                "escalation_decision": esc,
                "escalation_reason": reason,
                "human_eval_scores": {
                    "relevance": 5,
                    "tone": 4,
                    "actionability": 4,
                    "safety": 5,
                }
            }
        })
        uid += 1

    # 3. Context Dependent Pool (25)
    for text, intent, esc, reason in context_dependent_pool:
        all_examples.append({
            "id": f"golden_{uid:03d}",
            "customer_message": text,
            "sampling_bucket": "thread_context_dependent",
            "ground_truth": {
                "intent": intent,
                "escalation_decision": esc,
                "escalation_reason": reason,
                "human_eval_scores": {
                    "relevance": 4,
                    "tone": 5,
                    "actionability": 4,
                    "safety": 5,
                }
            }
        })
        uid += 1

    # 4. Distribution Shift Pool (15)
    for text, intent, esc, reason in distribution_shift_pool:
        all_examples.append({
            "id": f"golden_{uid:03d}",
            "customer_message": text,
            "sampling_bucket": "distribution_shift",
            "ground_truth": {
                "intent": intent,
                "escalation_decision": esc,
                "escalation_reason": reason,
                "human_eval_scores": {
                    "relevance": 5,
                    "tone": 5,
                    "actionability": 5,
                    "safety": 5,
                }
            }
        })
        uid += 1

    # Write golden_set.jsonl
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Generate 50 judge calibration instances (pairs of human score vs test data)
    calibration_set = []
    for ex in all_examples[:50]:
        h_scores = ex["ground_truth"]["human_eval_scores"]
        calibration_set.append({
            "id": ex["id"],
            "customer_message": ex["customer_message"],
            "intent": ex["ground_truth"]["intent"],
            "human_scores": h_scores,
        })

    with open(JUDGE_CALIBRATION_PATH, "w", encoding="utf-8") as f:
        for item in calibration_set:
            f.write(json.dumps(item) + "\n")

    print(f"Successfully generated {len(all_examples)} golden examples in {GOLDEN_SET_PATH}")
    print(f"Generated 50 calibration instances in {JUDGE_CALIBRATION_PATH}")


if __name__ == "__main__":
    create_golden_evaluation_dataset()

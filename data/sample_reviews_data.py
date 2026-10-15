"""
Realistic E-Commerce Products & Customer Reviews Dataset Generator.
Generates:
1. Product Catalog (50 items across Tech, Audio, Footwear, Home & Office)
2. Customer Reviews (1,500 rich customer reviews with sentiments, ratings, aspects)
3. Golden Evaluation Set (100 factual queries with labeled ground-truth review citations)
"""

import json
import os
import random
from typing import Dict, List
import numpy as np

random.seed(42)
np.random.seed(42)

PRODUCT_TEMPLATES = [
    ("PROD_001", "AeroPulse ANC Wireless Headphones", "Audio", 249.99, [
        ("battery", "The battery consistently delivers 34 hours of playback with ANC active. Flew from NYC to Tokyo and still had 40% left."),
        ("comfort", "The memory foam ear cushions are lightweight and breathable, no pressure on ears even after 8 hours of continuous work."),
        ("anc", "Active Noise Cancellation eliminates airplane cabin drone and coffee shop chatter with impressive transparency mode."),
        ("sound", "Sound signature has punchy sub-bass with crisp treble; EQ in companion app lets you dial in custom vocal curves."),
        ("mic", "The quad beamforming microphones isolate voice well on Zoom calls, though in strong wind outside callers noticed slight muffled audio."),
    ]),
    ("PROD_002", "ApexPro 16 Ultra Laptop", "Computers", 1499.99, [
        ("battery", "Battery runtime under code compilation and Docker containers is around 9.5 hours; fast charges to 80% in 45 minutes."),
        ("display", "The 16-inch 120Hz Mini-LED display reaches 1000 nits peak brightness with accurate DCI-P3 color coverage."),
        ("keyboard", "Tactile scissor switches with 1.5mm travel make typing code effortless; glass trackpad has perfect palm rejection."),
        ("thermals", "Dual vapor chamber cooling keeps fans completely silent during web browsing, humming quietly only under full Blender renders."),
        ("ports", "Generous I/O with two Thunderbolt 4 ports, HDMI 2.1, full-size SD card slot, and 3.5mm headphone jack."),
    ]),
    ("PROD_003", "BaristaTouch Thermal Espresso Machine", "Appliances", 699.99, [
        ("steam", "The dual boiler allows simultaneous espresso extraction and powerful microfoam steaming in under 45 seconds."),
        ("grinder", "Integrated conical burr grinder has 30 precise grind settings; single-dosing bellows prevents retention."),
        ("cleaning", "Cleaning the commercial-grade stainless steam wand is quick: a 2-second purge and damp microfiber wipe keeps it spotless."),
        ("heating", "ThermoJet heating block reaches optimal 93C brewing temperature in just 3 seconds from standby."),
        ("durability", "Solid brushed stainless steel construction feels commercial-grade with zero plastic in the pressurized flow path."),
    ]),
    ("PROD_004", "CloudStride Carbon Road Running Shoes", "Footwear", 179.99, [
        ("sizing", "Runs slightly snug in the toe box. Runners with wider feet should order a half-size up for marathon distances."),
        ("cushioning", "Supercritical PEBA midsole foam combined with curved carbon-fiber plate delivers responsive energy rebound."),
        ("durability", "Outsole Continental rubber shows minimal wear after 250 miles of road running; upper engineered mesh holds shape."),
        ("weight", "Ultralight build at just 7.4 ounces (men size 9.5), feels practically weightless on speed and tempo sessions."),
        ("stability", "Best suited for neutral runners; heel counter provides locked-in lockdown but lacks medial arch posting."),
    ]),
    ("PROD_005", "ErgoFlow Lumbar Mesh Office Chair", "Office", 389.99, [
        ("lumbar", "Self-adjusting dynamic lumbar support tracks posture shifts automatically, relieving lower back tightness."),
        ("mesh", "Breathable Korean elastomeric mesh prevents heat accumulation during humid summer workdays."),
        ("armrests", "4D armrests adjust in height, width, depth, and pivot 20 degrees inward for comfortable keyboard typing."),
        ("recline", "Synchronous tilt mechanism reclines smoothly up to 135 degrees with 4 lockable tilt angle positions."),
        ("assembly", "Shipped with clear visual manual and hex tool; unboxing and full assembly took roughly 18 minutes."),
    ]),
]

# Generate 45 additional synthetic products
CATEGORIES = ["Audio", "Computers", "Appliances", "Footwear", "Office", "Smart Home", "Fitness"]
ADJECTIVES = ["Quantum", "Veloce", "Titan", "Zenith", "Pulse", "Hydro", "Nova", "Starlight", "Core", "Swift"]
NOUNS = ["Earbuds", "Monitor", "Smartwatch", "Blender", "Air Purifier", "Backpack", "Standing Desk", "Mechanical Keyboard", "Camera", "Drone"]

for p_num in range(6, 51):
    pid = f"PROD_{p_num:03d}"
    cat = random.choice(CATEGORIES)
    name = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(['Pro', 'Max', 'Plus', 'SE', 'Elite'])}"
    price = round(random.uniform(39.99, 899.99), 2)
    aspects = [
        ("performance", f"Delivers exceptional {cat.lower()} performance with high reliability and intuitive controls."),
        ("battery", f"Efficient power management provides extended usage with fast recharging via USB-C."),
        ("build", f"Engineered with premium aerospace-grade aluminum and durable textured finish."),
        ("value", f"Offers competitive features matching flagship competitors at a considerably lower price point."),
    ]
    PRODUCT_TEMPLATES.append((pid, name, cat, price, aspects))


def generate_dataset(output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)

    products = []
    reviews = []
    golden_eval = []

    review_id_counter = 1000
    query_id_counter = 1

    for pid, pname, pcat, price, aspects in PRODUCT_TEMPLATES:
        products.append({
            "product_id": pid,
            "name": pname,
            "category": pcat,
            "price": price,
            "aspect_summaries": [asp[0] for asp in aspects],
        })

        # Generate 30 reviews per product (total 50 * 30 = 1,500 reviews)
        product_rev_ids = []
        for r_idx in range(30):
            review_id_counter += 1
            rid = f"REV_{review_id_counter}"
            product_rev_ids.append(rid)

            # Pick 1 or 2 focal aspects
            chosen_aspects = random.sample(aspects, k=random.choice([1, 2]))
            rating = int(np.random.choice([5, 4, 3, 2, 1], p=[0.55, 0.25, 0.10, 0.06, 0.04]))
            verified = bool(random.random() > 0.10)

            # Build realistic review text
            sentences = [f"I have been testing the {pname} for the past month."]
            for asp_key, asp_text in chosen_aspects:
                if rating >= 4:
                    sentences.append(f"Regarding {asp_key}: {asp_text}")
                elif rating == 3:
                    sentences.append(f"On the {asp_key} front, it's decent but has caveats. {asp_text}")
                else:
                    sentences.append(f"Disappointed with {asp_key}. It failed to match expectations.")

            rev_text = " ".join(sentences)
            title = f"{'Excellent' if rating >= 4 else 'Mixed feelings about'} {pname}"

            reviews.append({
                "review_id": rid,
                "product_id": pid,
                "rating": rating,
                "verified_purchase": verified,
                "title": title,
                "review_text": rev_text,
                "aspects_mentioned": [a[0] for a in chosen_aspects],
            })

        # Generate 2 golden evaluation queries per product (50 * 2 = 100 evaluation pairs)
        for q_idx in range(2):
            target_aspect = aspects[q_idx % len(aspects)]
            aspect_name, factual_claim = target_aspect

            # Find matching reviews that explicitly discussed this aspect
            matching_rids = [
                r["review_id"] for r in reviews
                if r["product_id"] == pid and aspect_name in r["aspects_mentioned"] and r["rating"] >= 4
            ]

            query_prompts = {
                "battery": f"How long does the battery last on the {pname} during heavy travel or work?",
                "comfort": f"Is the {pname} comfortable for long sessions or wearing all day?",
                "anc": f"How well does the active noise cancellation perform on the {pname}?",
                "sound": f"What is the sound quality and audio signature like for {pname}?",
                "sizing": f"Does the {pname} fit true to size or should I size up?",
                "cleaning": f"How easy is it to clean and maintain the {pname}?",
                "display": f"How bright and color-accurate is the screen on {pname}?",
                "lumbar": f"Does the {pname} provide good lower back and lumbar support?",
                "steam": f"How fast does the {pname} steam milk and heat up?",
                "performance": f"What is the overall performance and reliability of {pname}?",
                "build": f"What materials is the {pname} made of and how durable is the build?",
                "value": f"Is {pname} good value for money compared to alternatives?",
            }

            q_text = query_prompts.get(aspect_name, f"What do customers say about the {aspect_name} of {pname}?")

            golden_eval.append({
                "query_id": f"EVAL_{query_id_counter:03d}",
                "product_id": pid,
                "product_name": pname,
                "aspect": aspect_name,
                "query": q_text,
                "ground_truth_relevant_review_ids": matching_rids[:5],
                "reference_answer": factual_claim,
            })
            query_id_counter += 1

    # Save to JSON
    with open(os.path.join(output_dir, "products.json"), "w") as f:
        json.dump(products, f, indent=2)

    with open(os.path.join(output_dir, "reviews.json"), "w") as f:
        json.dump(reviews, f, indent=2)

    with open(os.path.join(output_dir, "golden_evaluation_set.json"), "w") as f:
        json.dump(golden_eval, f, indent=2)

    print(f"Generated RAG Dataset:")
    print(f"  Products: {len(products)}")
    print(f"  Reviews:  {len(reviews):,} customer reviews")
    print(f"  Evaluation Pairs: {len(golden_eval)} labeled test queries")
    return products, reviews, golden_eval


if __name__ == "__main__":
    generate_dataset()

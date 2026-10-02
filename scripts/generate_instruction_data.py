"""Generate synthetic 311 complaint -> dispatch ticket pairs.

Writes data/train.jsonl (340), data/val.jsonl (30), data/test.jsonl (30).
Each line: {"id", "complaint", "ticket": {category, urgency, location, summary, address}, "meta": {...}}

Address cases (by construction, never mixed):
  full_address  (70%) - complaint contains a street address; ticket.address copies it exactly.
  no_location   (15%) - complaint has no location at all; ticket.address and ticket.location are null.
  landmark_only (15%) - complaint names only a landmark; ticket.address is null, landmark goes in location.

Addresses and landmarks are "protected" text: the typo/slang/casing noise never touches them,
so the ticket can quote them word-for-word from the complaint.

Usage: python scripts/generate_instruction_data.py [--seed 311]
"""

import argparse
import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_TOTAL = 400
CASE_COUNTS = {"full_address": 280, "no_location": 60, "landmark_only": 60}
# Stratified so val and test both contain every address case.
SPLIT_COUNTS = {
    "val": {"full_address": 21, "no_location": 4, "landmark_only": 5},
    "test": {"full_address": 21, "no_location": 5, "landmark_only": 4},
}

# ---------------------------------------------------------------------------
# Issue bank. Urgency is decided by what the issue IS, never by the resident's tone.
# "say": (how a resident might describe it, [summaries that state ONLY facts in that phrasing]).
#        Summaries never borrow from "details" (which are optional), so they never invent facts.
# "details": optional extra color consistent with the urgency.
# ---------------------------------------------------------------------------
ISSUES = {
    "pothole": [
        dict(urgency="medium",
             say=[("there's a huge pothole", ["Large pothole in roadway.", "Big pothole reported; needs patching."]),
                  ("big pothole in the road", ["Large pothole in the road.", "Roadway pothole needs repair."]),
                  ("there is a pothole that keeps getting bigger", ["Growing pothole in roadway.", "Pothole increasing in size; patch requested."]),
                  ("massive hole in the street", ["Large hole in street surface.", "Major pothole in street."]),
                  ("pothole in the middle of the lane", ["Pothole in center of travel lane.", "Mid-lane pothole reported."])],
             details=["Cars keep swerving around it.", "It's been there for weeks.",
                      "It fills with water every time it rains so you can't see how deep it is."]),
        dict(urgency="high",
             say=[("a pothole blew out my tire", ["Pothole caused tire blowout; safety hazard.", "Tire blown out by pothole."]),
                  ("I hit a pothole and bent my rim", ["Vehicle rim bent by pothole.", "Pothole damaged resident's wheel rim."]),
                  ("there's a pothole so deep a car got stuck in it", ["Pothole deep enough to trap a vehicle.", "Car stuck in deep pothole; hazard."]),
                  ("pothole just wrecked my front tire", ["Pothole damaged resident's front tire.", "Tire damage from pothole; urgent repair needed."])],
             details=["Somebody is going to get hurt.", "A guy on a motorcycle almost wiped out yesterday.",
                      "My mechanic said the rim is ruined."]),
        dict(urgency="low",
             say=[("small pothole starting to form", ["Small pothole beginning to form.", "Early-stage pothole reported."]),
                  ("the road is getting a few little potholes", ["Several small potholes developing.", "Minor potholes forming on road surface."]),
                  ("minor pothole near the curb", ["Minor pothole near curb.", "Small curbside pothole."])],
             details=["Not urgent but figured I'd report it.", "Would be good to fix before winter."]),
    ],
    "streetlight": [
        dict(urgency="medium",
             say=[("the streetlight is out", ["Streetlight out.", "Non-working streetlight reported."]),
                  ("street light hasn't worked in a week", ["Streetlight out for about a week.", "Week-long streetlight outage."]),
                  ("the light pole is totally dark", ["Street lamp completely dark.", "Light pole not illuminating."]),
                  ("streetlight burned out", ["Burned-out streetlight.", "Street lamp needs replacement."])],
             details=["It's really dark walking home.", "I've noticed it for a few nights now."]),
        dict(urgency="low",
             say=[("streetlight keeps flickering on and off", ["Flickering streetlight.", "Streetlight cycling on and off."]),
                  ("the street light buzzes and flickers all night", ["Buzzing, flickering streetlight overnight.", "Street lamp flickers and buzzes at night."]),
                  ("light pole flickers", ["Light pole flickering.", "Intermittent street lamp."])],
             details=["It's more annoying than anything.", "It shines right in my bedroom window when it flickers."]),
        dict(urgency="high",
             say=[("every streetlight on the block is out", ["All streetlights on block out.", "Block-wide streetlight outage."]),
                  ("the whole street is pitch black at night", ["Entire street without lighting at night.", "Street completely dark at night; lighting outage."]),
                  ("all the street lights went out", ["Multiple streetlights out.", "Widespread streetlight outage reported."])],
             details=["Kids walk home from the bus stop in the dark.", "Someone got their car broken into last night.",
                      "You literally cannot see the crosswalk."]),
        dict(urgency="high",
             say=[("a light pole got hit by a car and is leaning", ["Light pole leaning after vehicle strike.", "Car-damaged light pole leaning; hazard."]),
                  ("there are wires hanging out of the bottom of a light pole", ["Exposed wires at base of light pole.", "Light pole with hanging wiring; electrical hazard."]),
                  ("the streetlight pole is cracked and tilting", ["Cracked, tilting streetlight pole.", "Streetlight pole damaged and leaning."])],
             details=["It looks like it could fall over.", "Kids were poking at the wires earlier."]),
    ],
    "graffiti": [
        dict(urgency="low",
             say=[("someone tagged the wall with graffiti", ["Graffiti tag on wall.", "Wall tagged; removal requested."]),
                  ("spray paint all over the fence", ["Spray paint on fence.", "Fence vandalized with spray paint."]),
                  ("there's graffiti on the bus shelter", ["Graffiti on bus shelter.", "Bus shelter tagged."]),
                  ("new graffiti showed up overnight", ["New graffiti appeared overnight.", "Overnight graffiti reported."])],
             details=["It's an eyesore.", "It keeps coming back after it gets painted over."]),
        dict(urgency="medium",
             say=[("someone spray painted a racial slur", ["Racial slur spray-painted; priority removal.", "Hate graffiti (racial slur) reported."]),
                  ("graffiti with obscene drawings right where kids can see", ["Obscene graffiti visible to children.", "Graffiti with obscene drawings in view of kids."]),
                  ("hateful graffiti on the wall", ["Hateful graffiti on wall.", "Offensive graffiti; prompt removal requested."])],
             details=["My kids saw it on the way to school.", "It's really upsetting for the neighborhood."]),
        dict(urgency="high",
             say=[("the stop sign is covered in graffiti and you can't read it", ["Stop sign unreadable due to graffiti; traffic hazard.", "Graffiti obscuring stop sign."]),
                  ("someone painted over the stop sign", ["Stop sign painted over; traffic safety hazard.", "Defaced stop sign."])],
             details=["Cars are rolling right through the intersection.", "Almost saw a crash this morning."]),
    ],
    "missed_trash_pickup": [
        dict(urgency="low",
             say=[("trash didn't get picked up this week", ["Missed trash pickup this week.", "Trash not collected this week."]),
                  ("the garbage truck skipped our street", ["Garbage truck skipped street.", "Street missed on garbage route."]),
                  ("recycling wasn't collected today", ["Recycling not collected today.", "Missed recycling pickup."])],
             details=["Bins were out the night before like always.", "Everyone else on the street is wondering too."]),
        dict(urgency="medium",
             say=[("trash hasn't been picked up in two weeks", ["Trash uncollected for two weeks.", "Two weeks of missed trash pickup."]),
                  ("recycling was missed for the third week in a row", ["Recycling missed three weeks in a row.", "Third consecutive missed recycling pickup."]),
                  ("nobody has collected our garbage in weeks", ["Garbage uncollected for several weeks.", "Multi-week missed garbage collection."])],
             details=["The bins are overflowing.", "We've called before and nothing changes."]),
        dict(urgency="high",
             say=[("garbage is piling up and now there are rats", ["Accumulated garbage attracting rats.", "Rats around piled-up garbage; health hazard."]),
                  ("trash bags everywhere and the rats are out in daylight", ["Rats seen in daylight among trash bags.", "Scattered trash bags with daytime rat activity."])],
             details=["It smells horrible.", "My kid saw a rat the size of a cat."]),
    ],
    "noise": [
        dict(urgency="medium",
             say=[("the neighbors are blasting music every night", ["Nightly loud music from neighbors.", "Neighbors playing loud music every night."]),
                  ("there's a loud party going until 3am", ["Loud party lasting until 3am.", "Late-night party noise."]),
                  ("someone is playing music so loud the windows shake", ["Music loud enough to shake windows.", "Extremely loud music complaint."])],
             details=["I have to work at 6.", "It's been going on for hours."]),
        dict(urgency="medium",
             say=[("a construction crew starts jackhammering at 5am", ["Jackhammering starting at 5am.", "Early-morning construction noise."]),
                  ("they're doing construction on a Sunday morning", ["Sunday morning construction noise.", "Construction work on Sunday morning."]),
                  ("construction noise way before allowed hours", ["Construction noise before permitted hours.", "Possible construction hours violation."])],
             details=["I thought there were rules about this.", "The whole block is awake."]),
        dict(urgency="low",
             say=[("a dog barks all day while the owners are at work", ["Dog barking all day while owners away.", "Barking dog left alone during the day."]),
                  ("the neighbor's dog won't stop barking", ["Neighbor's dog barking constantly.", "Persistent dog barking."])],
             details=["It starts at like 8am and goes all day.", "I work from home so I hear it constantly."]),
        dict(urgency="medium",
             say=[("a car alarm has been going off for an hour", ["Car alarm sounding for an hour.", "Hour-long car alarm."]),
                  ("some car alarm keeps going off all night", ["Car alarm going off repeatedly overnight.", "Overnight car alarm disturbance."])],
             details=["Nobody is coming to turn it off.", "It stops and starts again every ten minutes."]),
    ],
    "water_leak": [
        dict(urgency="high",
             say=[("water is gushing out of the street", ["Water gushing from street.", "Major water leak from roadway."]),
                  ("a water main broke and the road is flooding", ["Water main break flooding road.", "Road flooding from broken water main."]),
                  ("there's water shooting up out of the pavement", ["Water shooting up through pavement.", "Water erupting from pavement; urgent."])],
             details=["It's starting to get into people's basements.", "Cars are driving through like a river."]),
        dict(urgency="medium",
             say=[("the fire hydrant is leaking nonstop", ["Fire hydrant leaking continuously.", "Nonstop hydrant leak."]),
                  ("a hydrant has been dripping for days", ["Hydrant dripping for several days.", "Multi-day hydrant drip."])],
             details=["There's a big puddle that never goes away.", "Seems like a waste of water."]),
        dict(urgency="low",
             say=[("there's a small leak bubbling up near the curb", ["Small leak bubbling near curb.", "Minor curbside water leak."]),
                  ("there's a puddle that never dries up, I think it's a leak", ["Persistent puddle; suspected leak.", "Possible leak causing standing water."])],
             details=["It's not a lot of water.", "It's been there even when it hasn't rained."]),
        dict(urgency="high",
             say=[("sewage is coming up out of the storm drain", ["Sewage overflowing from storm drain.", "Storm drain backing up with sewage; health hazard."]),
                  ("there's raw sewage smell and dirty water coming up from a drain", ["Sewage odor and dirty water from drain.", "Possible sewer backup at drain."])],
             details=["It smells awful.", "Kids play right near there."]),
    ],
    "abandoned_vehicle": [
        dict(urgency="low",
             say=[("a car hasn't moved in a month", ["Car unmoved for a month.", "Possibly abandoned car, stationary one month."]),
                  ("an old van with flat tires has been sitting there for weeks", ["Van with flat tires parked for weeks.", "Possibly abandoned van with flat tires."]),
                  ("there's a car with expired tags that never moves", ["Unmoving car with expired tags.", "Car with expired registration left parked."])],
             details=["Leaves are piling up under it.", "Nobody on the street knows whose it is."]),
        dict(urgency="medium",
             say=[("an abandoned truck is blocking my driveway", ["Abandoned truck blocking driveway.", "Driveway obstructed by abandoned truck."]),
                  ("a stripped car is blocking half the lane", ["Stripped car blocking half a lane.", "Partially blocked lane from stripped vehicle."]),
                  ("a junk car is blocking the fire lane", ["Junk car blocking fire lane.", "Fire lane obstructed by junk vehicle."])],
             details=["I can't get my car out.", "Other cars have to go around it."]),
        dict(urgency="medium",
             say=[("an abandoned car with smashed windows is leaking oil", ["Abandoned car with smashed windows leaking oil.", "Vandalized vehicle leaking oil."]),
                  ("there's a burned out car that's been dumped", ["Burned-out car dumped.", "Dumped burned-out vehicle."])],
             details=["There's glass all over the ground.", "Nobody has come to tow it."]),
    ],
    "sidewalk": [
        dict(urgency="low",
             say=[("the sidewalk is cracked and uneven", ["Cracked, uneven sidewalk.", "Uneven sidewalk surface reported."]),
                  ("tree roots pushed the sidewalk up", ["Sidewalk lifted by tree roots.", "Root-heaved sidewalk."]),
                  ("the sidewalk slabs are all lifted", ["Lifted sidewalk slabs.", "Raised, uneven sidewalk slabs."])],
             details=["It's hard with a stroller.", "Been like that for a long time."]),
        dict(urgency="high",
             say=[("my mom tripped on the broken sidewalk and hurt her wrist", ["Pedestrian injured after tripping on broken sidewalk.", "Broken sidewalk caused fall and wrist injury."]),
                  ("someone fell on the busted sidewalk and needed stitches", ["Fall on broken sidewalk required stitches.", "Damaged sidewalk caused injury requiring stitches."])],
             details=["This needs to be fixed before someone else gets hurt.", "It's a lawsuit waiting to happen."]),
        dict(urgency="medium",
             say=[("the sidewalk is completely blocked so wheelchairs have to go in the street", ["Blocked sidewalk forcing wheelchairs into street.", "Sidewalk obstruction; accessibility issue."]),
                  ("a whole chunk of sidewalk is missing", ["Section of sidewalk missing.", "Missing sidewalk segment."])],
             details=["There's a man in a wheelchair who uses it every day.", "People are walking in the road."]),
    ],
    "tree_damage": [
        dict(urgency="high",
             say=[("a tree fell across the road", ["Fallen tree across road.", "Tree down blocking roadway."]),
                  ("a big tree came down in the storm and is blocking the street", ["Storm-downed tree blocking street.", "Large fallen tree obstructing street."])],
             details=["Nobody can get through.", "It took down part of a fence too."]),
        dict(urgency="high",
             say=[("a tree branch is resting on the power lines", ["Branch resting on power lines.", "Tree limb on power lines; electrical hazard."]),
                  ("a limb fell onto the power lines and is sparking", ["Fallen limb on power lines, sparking.", "Sparking power lines with limb on them."])],
             details=["The lights in our house keep flickering.", "I can hear it crackling."]),
        dict(urgency="medium",
             say=[("a huge branch cracked and is hanging over the sidewalk", ["Cracked branch hanging over sidewalk.", "Broken overhead branch above sidewalk."]),
                  ("a dead tree is leaning toward the houses", ["Dead tree leaning toward homes.", "Leaning dead tree near houses."])],
             details=["It moves when the wind blows.", "People walk under it all day."]),
        dict(urgency="low",
             say=[("the tree needs trimming, branches are scraping cars", ["Tree trimming needed; branches scraping cars.", "Low branches hitting vehicles."]),
                  ("overgrown branches are hanging low over the street", ["Overgrown branches hanging low over street.", "Tree trimming requested over roadway."])],
             details=["Not an emergency.", "The trucks hit it every time they come by."]),
    ],
    "animal_control": [
        dict(urgency="high",
             say=[("a loose pit bull is chasing people", ["Loose pit bull chasing people.", "Aggressive loose dog chasing pedestrians."]),
                  ("an aggressive dog is roaming the street and it bit someone", ["Aggressive roaming dog; bite reported.", "Dog bite by loose aggressive dog."]),
                  ("a big dog is off leash and lunging at kids", ["Off-leash dog lunging at children.", "Large loose dog threatening kids."])],
             details=["People are scared to go outside.", "Nobody knows who the owner is."]),
        dict(urgency="medium",
             say=[("there's a dead raccoon in the road", ["Dead raccoon in roadway.", "Raccoon carcass removal requested."]),
                  ("a dead deer is on the side of the road", ["Dead deer on roadside.", "Deer carcass pickup requested."]),
                  ("a dead cat has been lying there since yesterday", ["Dead cat present since yesterday.", "Cat carcass removal requested."])],
             details=["It's starting to smell.", "Nobody has picked it up."]),
        dict(urgency="low",
             say=[("a stray cat is hanging around", ["Stray cat in area.", "Stray cat reported."]),
                  ("there's a stray dog that seems friendly but lost", ["Friendly stray dog, appears lost.", "Lost-looking stray dog reported."])],
             details=["It doesn't have a collar.", "It's been around for a couple days."]),
        dict(urgency="medium",
             say=[("a raccoon is living in the storm drain", ["Raccoon living in storm drain.", "Wildlife denning in storm drain."]),
                  ("there's a coyote walking around in broad daylight", ["Coyote seen in daylight.", "Daytime coyote sighting."])],
             details=["We have small kids and pets.", "It didn't seem scared of people at all."]),
    ],
}

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
STREETS = ["Maple", "Oak", "Cedar", "Elm", "Pine", "Washington", "Lincoln", "Jefferson", "Park", "Lake",
           "Hill", "River", "Sunset", "Highland", "Franklin", "Madison", "Chestnut", "Walnut", "Spruce",
           "Willow", "Grant", "Jackson", "Adams", "Monroe", "Main", "Church", "Mill", "Spring", "Center",
           "1st", "2nd", "3rd", "5th", "7th", "9th", "12th", "Martin Luther King Jr", "Cherry", "Prospect",
           "Bayview", "Fairview", "Magnolia", "Orchard", "Hawthorne", "Delaware", "Kingsley"]
SUFFIXES = [("St", "Street"), ("Ave", "Avenue"), ("Rd", "Road"), ("Blvd", "Boulevard"), ("Dr", "Drive"),
            ("Ln", "Lane"), ("Ct", "Court"), ("Pl", "Place")]
DIRECTIONS = ["", "", "", "", "N ", "S ", "E ", "W "]

STORES = ["Walgreens", "CVS", "Safeway", "Kroger", "Target", "Dollar General", "7-Eleven", "Shell station",
          "McDonald's", "Home Depot", "Starbucks", "Wells Fargo", "Taco Bell", "AutoZone"]
SCHOOLS = ["Lincoln Elementary", "Roosevelt High", "Jefferson Middle School", "the community college",
           "St. Mary's school", "Garfield Elementary"]
PARKS = ["Riverside Park", "Memorial Park", "Oak Hollow Park", "the dog park", "Lakeview Park", "Veterans Park"]
CIVIC = ["public library", "fire station", "post office", "rec center", "senior center", "DMV"]
BARE_STREETS = ["7th", "Main", "Elm", "Broadway", "Washington", "3rd", "Lincoln", "Grand", "Park"]
LANDMARKS = [
    "behind the {store} on {street}", "in the parking lot of the {store} on {street}", "across from {school}",
    "near the entrance to {park}", "by the playground at {park}", "next to the bus stop outside {school}",
    "under the {street} bridge", "on the trail behind {park}", "in the alley behind the {store}",
    "right outside the {civic}", "across the street from the {civic}", "at the corner by the {store}",
    "next to the {store} on {street}", "by the basketball courts at {park}", "near the {civic} on {street}",
    "at the light by the {store}",
]
VAGUE = ["on my street", "in my neighborhood", "near my house", "down the road", "around here", "on our block"]


def make_address(rng):
    num = rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 9999)])
    abbr, full = rng.choice(SUFFIXES)
    addr = f"{num} {rng.choice(DIRECTIONS)}{rng.choice(STREETS)} {full if rng.random() < 0.35 else abbr}"
    return addr.lower() if rng.random() < 0.15 else addr


def make_landmark(rng):
    lm = rng.choice(LANDMARKS).format(store=rng.choice(STORES), street=rng.choice(BARE_STREETS),
                                      school=rng.choice(SCHOOLS), park=rng.choice(PARKS), civic=rng.choice(CIVIC))
    return lm.lower() if rng.random() < 0.3 else lm


# ---------------------------------------------------------------------------
# Tone pieces. Flags feed into the summary ("repeat" / "callback").
# ---------------------------------------------------------------------------
ANGRY_OPEN = [("This is ridiculous.", None), ("I am so sick of this.", None), ("Are you kidding me.", None),
              ("Third time calling about this.", "repeat"), ("Why do I even pay taxes.", None),
              ("Nobody in this city ever fixes anything.", None), ("I reported this LAST MONTH.", "repeat"),
              ("Unbelievable.", None), ("Hey city, wake up.", None), ("Still not fixed!!", "repeat")]
ANGRY_CLOSE = [("Fix it NOW.", None), ("Do your job.", None), ("If someone gets hurt it's on you.", None),
               ("I want a call back today.", "callback"), ("If this isn't fixed I'm calling the news.", None),
               ("Ridiculous.", None), ("Somebody call me back.", "callback"), ("Get it done.", None)]
POLITE_OPEN = [("Hi there,", None), ("Hello,", None), ("Good morning.", None),
               ("Hi, I hope this is the right place to report this.", None), ("Sorry to bother you.", None),
               ("Hello 311.", None), ("Hi! Just wanted to let someone know about something.", None),
               ("Good evening, I'd like to report a problem.", None), ("I think I reported this before but", "repeat")]
POLITE_CLOSE = [("Thank you so much!", None), ("Thanks for your help.", None), ("Appreciate it.", None),
                ("Please let me know if you need anything else.", None),
                ("Could someone call me back when it's scheduled? Thanks.", "callback"),
                ("Have a great day.", None), ("Thanks in advance.", None), ("Really appreciate all you do.", None)]
RAMBLE = ["I've lived here 20 years and never seen it this bad.", "My husband says I should just let it go but",
          "So anyway I was walking my dog this morning.", "Not sure if this is the right number honestly.",
          "My neighbor told me to call 311 about it.", "We pay a lot in taxes here you know.",
          "I was on my way to work when I noticed.", "Long story short.", "Okay so this is kind of a long story.",
          "My daughter was visiting and she said I should call.", "I don't usually complain about stuff but",
          "First of all I want to say I love this city.", "So yesterday I was coming back from the grocery store.",
          "I told the guy next door and he said call the city."]
SUMMARY_NOTES = {"repeat": ["Repeat report.", "Resident says this was reported before."],
                 "callback": ["Callback requested.", "Resident asked for a call back."]}

SLANG = {"you": "u", "your": "ur", "are": "r", "please": "pls", "because": "bc", "really": "rly",
         "going to": "gonna", "want to": "wanna", "about": "abt", "people": "ppl", "tonight": "tonite",
         "thanks": "thx", "probably": "prob", "don't know": "dunno", "kind of": "kinda", "something": "smth"}
MISSPELL = {"definitely": "definately", "because": "becuase", "neighbor": "nieghbor", "really": "realy",
            "until": "untill", "garbage": "garbge", "weird": "wierd", "street": "streat", "there's": "theres",
            "it's": "its", "can't": "cant", "don't": "dont", "isn't": "isnt", "hasn't": "hasnt", "I'm": "im",
            "wasn't": "wasnt", "week": "wk", "minutes": "mins"}

TONE_STYLE = {
    # typo rate per word, slang prob, lowercase prob, no-punctuation prob, run-on prob
    "angry":    dict(typo=0.04, slang=0.25, lower=0.25, nopunct=0.30, runon=0.30),
    "polite":   dict(typo=0.015, slang=0.05, lower=0.05, nopunct=0.05, runon=0.10),
    "rambling": dict(typo=0.05, slang=0.30, lower=0.55, nopunct=0.50, runon=0.85),
    "terse":    dict(typo=0.03, slang=0.30, lower=0.70, nopunct=0.70, runon=0.0),
}


def typo(word, rng):
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def noisify(text, st, rng):
    """Apply slang, misspellings, and typos to unprotected text only."""
    for k, v in SLANG.items():
        if rng.random() < st["slang"]:
            text = re.sub(rf"\b{re.escape(k)}\b", v, text, flags=re.I)
    for k, v in MISSPELL.items():
        if rng.random() < 0.35:
            text = re.sub(rf"(?<![\w']){re.escape(k)}(?![\w'])", v, text)
    return re.sub(r"[A-Za-z]+", lambda m: typo(m.group(), rng) if rng.random() < st["typo"] else m.group(), text)


def location_sentences(issue, case, loc, rng):
    """Combine the issue sentence with the location. Returns list of sentences; each is [(text, protected)]."""
    if case == "no_location":
        vague = rng.choice(VAGUE + [""] * 4)
        return [[(f"{issue} {vague}".strip(), False)]]
    r = rng.random()
    if case == "full_address":
        if r < 0.45:
            prep = rng.choice(["in front of", "outside", "at", "right by", "near", "across the street from",
                               "on the street outside", "on the corner at"])
            return [[(f"{issue} {prep} ", False), (loc, True)]]
        if r < 0.78:
            lead = rng.choice(["Address is ", "It's at ", "Location is ", "I live at ", "This is at ",
                               "The address is ", "It's right outside ", "Happening at "])
            return [[(issue, False)], [(lead, False), (loc, True)]]
        return [[(loc, True), (rng.choice([" - ", ": ", ", ", " "]) + issue, False)]]
    # landmark_only: phrase already carries its own preposition
    if r < 0.5:
        return [[(f"{issue} ", False), (loc, True)]]
    if r < 0.8:
        return [[(issue, False)], [(rng.choice(["It's ", "This is ", "Location: ", "It's right "]), False), (loc, True)]]
    return [[(loc, True), (rng.choice([", ", " - ", " "]) + issue, False)]]


def render(sentences, tone, rng):
    st = TONE_STYLE[tone]
    lower, nopunct = rng.random() < st["lower"], rng.random() < st["nopunct"]
    runon = rng.random() < st["runon"]
    out = []
    for si, sent in enumerate(sentences):
        pieces = []
        for pi, (text, protected) in enumerate(sent):
            if not protected:
                text = noisify(text, st, rng)
                if nopunct:
                    text = re.sub(r"[.,!?;:']", "", text)
                if lower:
                    text = text.lower()
                elif pi == 0 and text:
                    # Capitalize only at a real sentence start, not after "and then" / a comma.
                    starts_sentence = not out or out[-1].rstrip()[-1:] in ".!?" or (nopunct and not runon)
                    text = (text[0].upper() if starts_sentence else text[0].lower()) + text[1:]
                if tone == "angry" and rng.random() < 0.15:
                    text = text.upper()
            pieces.append(text)
        s = "".join(pieces).strip()
        if not s:
            continue
        last = si == len(sentences) - 1
        if runon and not last:
            s = re.sub(r"[.!]+$", "", s) + rng.choice([" and ", " so ", ", ", " ", " and then "])
        elif nopunct:
            s += " "
        else:
            if not re.search(r"[.!?,]$", s):
                s += "!" if tone == "angry" and rng.random() < 0.5 else "."
            if tone == "angry" and rng.random() < 0.2:
                s = s.rstrip(".") + "!!!"
            s += " "
        out.append(s)
    return "".join(out).strip()


def make_example(category, case, tone, rng):
    issue = rng.choice(ISSUES[category])
    say, say_summaries = rng.choice(issue["say"])
    loc = make_address(rng) if case == "full_address" else make_landmark(rng) if case == "landmark_only" else None
    if tone == "terse":
        say = re.sub(r"^(there's|there is|there are) (a |an )?", "", say)
    flags = set()

    sentences = []
    if tone == "angry" and rng.random() < 0.75:
        text, flag = rng.choice(ANGRY_OPEN); sentences.append([(text, False)]); flags.add(flag)
    if tone == "polite":
        text, flag = rng.choice(POLITE_OPEN); sentences.append([(text, False)]); flags.add(flag)
    if tone == "rambling":
        for text in rng.sample(RAMBLE, rng.choice([1, 2])):
            sentences.append([(text, False)])
    sentences += location_sentences(say, case, loc, rng)
    n_details = {"terse": 0 if rng.random() < 0.6 else 1, "rambling": 2}.get(tone, rng.choice([0, 1, 1]))
    for text in rng.sample(issue["details"], min(n_details, len(issue["details"]))):
        sentences.append([(text, False)])
    if tone == "rambling" and rng.random() < 0.5:
        sentences.append([(rng.choice(RAMBLE), False)])
    if tone == "angry" and rng.random() < 0.6:
        text, flag = rng.choice(ANGRY_CLOSE); sentences.append([(text, False)]); flags.add(flag)
    if tone == "polite" and rng.random() < 0.8:
        text, flag = rng.choice(POLITE_CLOSE); sentences.append([(text, False)]); flags.add(flag)

    summary = rng.choice(say_summaries)
    for flag in ("repeat", "callback"):
        if flag in flags:
            summary += " " + rng.choice(SUMMARY_NOTES[flag])

    ticket = {
        "category": category,
        "urgency": issue["urgency"],
        "location": loc,
        "summary": summary,
        "address": loc if case == "full_address" else None,
    }
    return {"complaint": render(sentences, tone, rng), "ticket": ticket,
            "meta": {"address_case": case, "tone": tone}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=311)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    categories = list(ISSUES) * (N_TOTAL // len(ISSUES))
    cases = [c for c, n in CASE_COUNTS.items() for _ in range(n)]
    tones = list(TONE_STYLE) * (N_TOTAL // len(TONE_STYLE))
    for lst in (categories, cases, tones):
        rng.shuffle(lst)

    examples, seen = [], set()
    for category, case, tone in zip(categories, cases, tones):
        for _ in range(100):
            ex = make_example(category, case, tone, rng)
            if ex["complaint"] not in seen:
                break
        else:
            raise RuntimeError(f"Could not generate a unique complaint for {category}/{case}/{tone}")
        seen.add(ex["complaint"])
        # Guarantees the core rule: the ticket's address is quoted exactly from the complaint.
        addr = ex["ticket"]["address"]
        assert addr is None or addr in ex["complaint"], ex
        examples.append(ex)

    for i, ex in enumerate(examples):
        ex["id"] = f"cd-{i + 1:04d}"
        ex["meta"]["seed"] = args.seed

    # Stratified split by address case.
    splits = {"train": [], "val": [], "test": []}
    for case in CASE_COUNTS:
        pool = [ex for ex in examples if ex["meta"]["address_case"] == case]
        rng.shuffle(pool)
        for split in ("val", "test"):
            n = SPLIT_COUNTS[split][case]
            splits[split] += pool[:n]
            pool = pool[n:]
        splits["train"] += pool
    for split, rows in splits.items():
        rng.shuffle(rows)

    DATA_DIR.mkdir(exist_ok=True)
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        with path.open("w") as f:
            for ex in rows:
                f.write(json.dumps({"id": ex["id"], "complaint": ex["complaint"], "ticket": ex["ticket"],
                                    "meta": ex["meta"]}) + "\n")
        print(f"wrote {len(rows):>3} examples -> {path}")


if __name__ == "__main__":
    main()

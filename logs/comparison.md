| test (30 examples) | base | sft_v1 | dpo_v1 | dpo_v2 |
|---|---|---|---|---|
| Valid JSON | 30/30 (100%) | 30/30 (100%) | 25/30 (83%) | 30/30 (100%) |
| Category correct | 5/30 (17%) | 27/30 (90%) | 20/30 (67%) | 27/30 (90%) |
| Urgency correct | 11/30 (37%) | 12/30 (40%) | 12/30 (40%) | 12/30 (40%) |
| Fabrication (no-address cases) | 0/9 (0%) | 0/9 (0%) | 2/9 (22%) | 2/9 (22%) |
|   of which invented | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) | 0/9 (0%) |
|   of which copied non-address | 0/9 (0%) | 0/9 (0%) | 2/9 (22%) | 2/9 (22%) |
| Location fabrication (all cases) | 5/30 (17%) | 1/30 (3%) | 0/30 (0%) | 1/30 (3%) |
| ANY fabrication (all cases) | 5/30 (17%) | 1/30 (3%) | 2/30 (7%) | 3/30 (10%) |
| False-null (address cases) | 17/21 (81%) | 0/21 (0%) | 0/21 (0%) | 0/21 (0%) |
| Address exact match | 4/21 (19%) | 19/21 (90%) | 16/21 (76%) | 19/21 (90%) |

| probe (20 examples) | base | sft_v1 | dpo_v1 | dpo_v2 |
|---|---|---|---|---|
| Valid JSON | 20/20 (100%) | 20/20 (100%) | 19/20 (95%) | 20/20 (100%) |
| Category correct | 6/20 (30%) | 17/20 (85%) | 14/20 (70%) | 16/20 (80%) |
| Urgency correct | 7/20 (35%) | 10/20 (50%) | 8/20 (40%) | 10/20 (50%) |
| Fabrication (no-address cases) | 0/10 (0%) | 2/10 (20%) | 2/10 (20%) | 2/10 (20%) |
|   of which invented | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
|   of which copied non-address | 0/10 (0%) | 2/10 (20%) | 2/10 (20%) | 2/10 (20%) |
| Location fabrication (all cases) | 3/20 (15%) | 0/20 (0%) | 0/20 (0%) | 0/20 (0%) |
| ANY fabrication (all cases) | 3/20 (15%) | 2/20 (10%) | 2/20 (10%) | 2/20 (10%) |
| False-null (address cases) | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |
| Address exact match | 0/10 (0%) | 8/10 (80%) | 9/10 (90%) | 9/10 (90%) |

```

== Probe complaints with NO address: what each model wrote ==

[probe-11] 'Called 3 times already!!! The light at the bus stop for route 42 has been out for 9 days. Case number 2024-118734. Nobody calls back.'
   base    address=None  location={'street_address': 'bus stop'}  category='pothole' urgency='high'
   sft_v1  address=None  location='at the bus stop for route 42'  category='streetlight' urgency='medium'
   dpo_v1  address=None  location='at the bus stop for route 42'  category='streetlight' urgency='high'
   dpo_v2  address=None  location='at the bus stop for route 42'  category='streetlight' urgency='medium'

[probe-12] 'dead possum on the bike path near mile marker 3 🤢🤢'
   base    address=None  location={'street_address': '2nd Avenue', 'suburb': 'Westminster'}  category='pothole' urgency='medium'  <-- FABRICATED
   sft_v1  address=None  location='on the bike path near mile marker 3'  category='animal_control' urgency='high'
   dpo_v1  address=None  location='by bike path'  category='animal_control' urgency='high'
   dpo_v2  address=None  location='on the bike path near mile marker 3'  category='animal_control' urgency='high'

[probe-13] 'To whom it may concern: I wish to report an abandoned motorcycle chained to the railing at the north entrance of Fairmount Station. It has not moved in approximately six weeks.'
   base    address=None  location={'street_address': 'Fairmount Station'}  category='abandoned_vehicle' urgency='high'
   sft_v1  address=None  location='north entrance of Fairmount Station'  category='abandoned_vehicle' urgency='medium'
   dpo_v1  address=None  location='north entrance of Fairmount Station'  category='abandoned_vehicle' urgency='low'
   dpo_v2  address=None  location='north entrance of Fairmount Station'  category='abandoned_vehicle' urgency='medium'

[probe-14] "[Voicemail] hi yeah it's apartment 4C calling, there's a dog upstairs barking literally all day every day. I'm at 555-0172 if you need me. thanks"
   base    address=None  location={'street_address': '555-0172'}  category='animal_control' urgency='high'
   sft_v1  address='555-0172'  location='555-0172'  category='animal_control' urgency='high'  <-- FABRICATED
   dpo_v1  address='555-0172'  location='555-0172'  category='animal_control' urgency='high'  <-- FABRICATED
   dpo_v2  address='555-0172'  location='555-0172'  category='animal_control' urgency='high'  <-- FABRICATED

[probe-15] "Our garbage hasn't been collected in 3 weeks and the bins are overflowing. We're in unit 12 of the townhomes. Please help."
   base    address=None  location={'street_address': 'Unit 12, Townhome #12'}  category='poor' urgency='high'
   sft_v1  address='unit 12 of the townhomes'  location='unit 12 of the townhomes'  category='missed_trash_pickup' urgency='medium'  <-- FABRICATED
   dpo_v1  address='12'  location='12'  category='missed_trash_pickup' urgency='high'  <-- FABRICATED
   dpo_v2  address='unit 12 of the townhomes'  location='unit 12 of the townhomes'  category='missed_trash_pickup' urgency='medium'  <-- FABRICATED

[probe-16] 'Hay graffiti con palabras muy feas en la pared del parque, near the swings. Mis hijos lo vieron y me preguntaron qué significa.'
   base    address=None  location={'street_address': 'Parque X, calle XX'}  category='pothole' urgency='medium'
   sft_v1  address=None  location='near the swings'  category='graffio' urgency='medium'
   dpo_v1  address=None  location='en la pared del parque, near the swings'  category='graffia' urgency='medium'
   dpo_v2  address=None  location='near the swings'  category='graffio' urgency='medium'

[probe-17] '• Problem: big hole in the road\n• Where: right after you get off Hwy 9 at exit 14, before the gas station\n• How long: about 2 weeks'
   base    address=None  location={'street_address': 'Hwy 9'}  category='pothole' urgency='high'
   sft_v1  address=None  location='right after you get off Hwy 9 at exit 14, before the gas station'  category='pothole' urgency='medium'
   dpo_v1  address=None  location='right after you get off Hwy 9 at exit 14, before the gas station'  category='streetlight' urgency='high'
   dpo_v2  address=None  location='right after you get off Hwy 9 at exit 14, before the gas station'  category='pothole' urgency='medium'

[probe-18] "The fire hydrant on my corner has been leaking since 6 AM. It's now 4 PM and it's still going."
   base    address=None  location={'street_address': 'corner of Main Street and your street', 'suburb': ''}  category='water_leak' urgency='high'
   sft_v1  address=None  location='on your corner'  category='water_leak' urgency='high'
   dpo_v1  address=None  location='in vicinity of the fire hydrant, null'  category='water_leak' urgency='high'
   dpo_v2  address=None  location='on your corner'  category='water_leak' urgency='high'

[probe-19] "This is the 2nd time I'm writing. A big branch from the city tree split during Saturday's storm and is hanging over the playground at the elementary school, maybe 20 feet up. Kids are under it every recess."
   base    address=None  location={'street_address': '2000 Elm St'}  category='pothole' urgency='high'  <-- FABRICATED
   sft_v1  address=None  location='over the playground at the elementary school'  category='tree_damage' urgency='high'
   dpo_v1  address=None  location='in the area of the reported damage, if provided'  category='tree_damage' urgency='high'
   dpo_v2  address=None  location='over the playground at the elementary school'  category='tree_damage' urgency='high'

[probe-20] 'sidewalk by the library is all broken up my grandmas walker got stuck in it yesterday and she couldnt get it out'
   base    address='null'  location='sidewalk by the library'  category='pothole' urgency='high'
   sft_v1  address=None  location='by the library'  category='sidewalk' urgency='medium'
   dpo_v1  address=None  location='by the library'  category='sidewalk' urgency='low'
   dpo_v2  address=None  location='by the library'  category='sidewalk' urgency='medium'

```
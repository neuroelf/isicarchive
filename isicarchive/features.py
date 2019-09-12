# from https://github.com/ImageMarkup/isic-archive/blob/master/isic_archive/models/masterFeatures.json

# order
# - circles / semicircles (24++)
# - dots (73++)
# - facial skin (163++)
# - globules / clods (215++)
#   - lacunae (260++)
# - lines (363++)
# - miscellaneous
# - nail lesions
# - network
# - pattern
#   - homogeneous pattern
# - regression structures
# - shiny white structures
# - structureless
#   - homogeneous pattern
# - vessels
# - volar lesions

master_features = [
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : NOS",
        "abbreviation": "Circ:NOS",
        "color": [120,120,120],
        "synonyms": ["Circles & Semicircles : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : Asymmetric follicular openings",
        "abbreviation": "Circ:Asym.O.",
        "synonyms": [
            "Circles & Semicircles : Asymmetric follicular openings",
            "Facial Skin : Asymmetric follicular openings"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : Brown",
        "abbreviation": "Circ:Brown",
        "color": [184,136,112],
        "synonyms": [
            "Circles & Semicircles : Brown",
            "Facial Skin : Brown circles"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : Circle within a circle",
        "abbreviation": "Circ:C.w/iC.",
        "color": [160,128,120],
        "synonyms": [
            "Circles & Semicircles : Circle within a circle",
            "Facial Skin : Circle within a circle",
            "Miscellaneous : Circle within a circle"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : Fish scale (mucosal lesions)",
        "abbreviation": "Circ:FishSc.",
        "synonyms": ["Circles & Semicircles : Fish scale (mucosal lesions)"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : Gray",
        "abbreviation": "Circ:Gray",
        "synonyms": ["Circles & Semicircles : Gray"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Circles & Semicircles : White",
        "abbreviation": "Circ:White",
        "color": [192,192,192],
        "synonyms": ["Circles & Semicircles : White"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : NOS",
        "abbreviation": "Dots:NOS",
        "color": [48,48,48],
        "synonyms": ["Dots : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Annular-granular pattern",
        "abbreviation": "Dots:An-gran",
        "synonyms": [
            "Dots : Annular-granular pattern",
            "Facial Skin : Annular-granular pattern",
            "Miscellaneous : Annular-granular pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Black",
        "abbreviation": "Dots:Black",
        "color": [48,40,24],
        "synonyms": ["Dots : Black"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Blue-gray",
        "abbreviation": "Dots:Bl-gray",
        "color": [8,64,128],
        "synonyms": ["Dots : Blue-gray"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Brown",
        "abbreviation": "Dots:Brown",
        "color": [76,48,40],
        "synonyms": ["Dots : Brown"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Granularity",
        "abbreviation": "Dots:Granul.",
        "synonyms": ["Dots : Granularity"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Irregular",
        "abbreviation": "Dots:Irreg.",
        "color": [96,56,56],
        "synonyms": ["Dots : Irregular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Linear",
        "abbreviation": "Dots:Linear",
        "synonyms": ["Dots : Linear"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Milia-like cysts",
        "abbreviation": "Dots:MiliaC.",
        "synonyms": [
            "Dots : Milia-like cysts",
            "Globules / Clods : Milia-like cysts",
            "Miscellaneous : Milia-like cysts, cloudy or starry"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Peppering",
        "abbreviation": "Dots:Pepper.",
        "synonyms": [
            "Dots : Peppering",
            "Regression structures : Peppering / Granularity"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Regular",
        "abbreviation": "Dots:Regular",
        "color": [72,48,48],
        "synonyms": ["Dots : Regular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Rosettes",
        "abbreviation": "Dots:Rosette",
        "synonyms": [
            "Dots : Rosettes",
            "Shiny white structures : Rosettes"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Targetoid",
        "abbreviation": "Dots:Target.",
        "synonyms": ["Dots : Targetoid"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Dots : Yellow",
        "abbreviation": "Dots:Yellow",
        "synonyms": ["Dots : Yellow"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Annular-granular pattern",
        "abbreviation": "Face:An-gran",
        "synonyms": [
            "Dots : Annular-granular pattern",
            "Facial Skin : Annular-granular pattern",
            "Miscellaneous : Annular-granular pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Asymmetric follicular openings",
        "abbreviation": "Face:Asym.O.",
        "synonyms": [
            "Circles & Semicircles : Asymmetric follicular openings",
            "Facial Skin : Asymmetric follicular openings"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Brown circles",
        "abbreviation": "Face:BrownC.",
        "synonyms": [
            "Circles & Semicircles : Brown",
            "Facial Skin : Brown circles"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Circle within a circle",
        "abbreviation": "Face:C.w/iC.",
        "synonyms": [
            "Circles & Semicircles : Circle within a circle",
            "Facial Skin : Circle within a circle",
            "Miscellaneous : Circle within a circle"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Pseudonetwork",
        "abbreviation": "Face:PseuNet",
        "synonyms": [
            "Facial Skin : Pseudonetwork",
            "Miscellaneous : Pseudonetwork",
            "Structureless : Pseudonetwork"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Facial Skin : Rhomboids / Zig-zag pattern",
        "abbreviation": "Face:Rhb/Zig",
        "synonyms": [
            "Facial Skin : Rhomboids / Zig-zag pattern",
            "Lines : Rhomboids (facial skin)",
            "Miscellaneous : Rhomboids / Zig-zag pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : NOS",
        "abbreviation": "G/C.:NOS",
        "color": [128,96,48],
        "synonyms": ["Globules / Clods : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Blue",
        "abbreviation": "G/C.:Blue",
        "color": [16,48,96],
        "synonyms": ["Globules / Clods : Blue"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Blue-gray ovoid nests",
        "abbreviation": "G/C.:Bl-grON",
        "color": [32,56,112],
        "synonyms": ["Globules / Clods : Blue-gray ovoid nests"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Cobblestone pattern",
        "abbreviation": "G/C.:CobblS.",
        "color": [48,48,104],
        "synonyms": ["Globules / Clods : Cobblestone pattern"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Comedo-like openings",
        "abbreviation": "G/C.:ComedoO",
        "color": [192,136,112],
        "synonyms": [
            "Globules / Clods : Comedo-like openings",
            "Miscellaneous : Comedo-like openings"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Concentric",
        "abbreviation": "G/C.:Concent",
        "color": [184,120,88],
        "synonyms": ["Globules / Clods : Concentric"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Irregular",
        "abbreviation": "G/C.:Irreg.",
        "color": [144,64,120],
        "synonyms": ["Globules / Clods : Irregular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Lacunae : NOS",
        "abbreviation": "G/C.:LacuNOS",
        "synonyms": [
            "Globules / Clods : Lacunae : NOS",
            "Miscellaneous : Lacunae : NOS"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Lacunae : Black",
        "abbreviation": "G/C.:LacuBlk",
        "color": [56,56,16],
        "synonyms": [
            "Globules / Clods : Lacunae : Black",
            "Miscellaneous : Lacunae : Black"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Lacunae : Blue",
        "abbreviation": "G/C.:LacuBlu",
        "synonyms": [
            "Globules / Clods : Lacunae : Blue",
            "Miscellaneous : Lacunae : Blue"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Lacunae : Red",
        "abbreviation": "G/C.:LacuRed",
        "color": [144,24,16],
        "synonyms": [
            "Globules / Clods : Lacunae : Red",
            "Miscellaneous : Lacunae : Red"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Lacunae : Red-purple",
        "abbreviation": "G/C.:LacuR-P",
        "color": [144,16,80],
        "synonyms": [
            "Globules / Clods : Lacunae : Red-purple",
            "Miscellaneous : Lacunae : Red-purple"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Leaflike area",
        "abbreviation": "G/C.:LeaflkA",
        "color": [72,48,64],
        "synonyms": [
            "Globules / Clods : Leaflike area",
            "Lines : Leaf like areas",
            "Miscellaneous : Leaflike area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Milia-like cysts",
        "abbreviation": "G/C.:MiliaC.",
        "color": [224,208,192],
        "synonyms": [
            "Dots : Milia-like cysts",
            "Globules / Clods : Milia-like cysts",
            "Miscellaneous : Milia-like cysts, cloudy or starry"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Milky red",
        "abbreviation": "G/C.:MilkRed",
        "color": [240,160,120],
        "synonyms": ["Globules / Clods : Milky red"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Regular",
        "abbreviation": "G/C.:Regular",
        "color": [152,80,72],
        "synonyms": ["Globules / Clods : Regular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Rim of brown globules",
        "abbreviation": "G/C.:RimBrwn",
        "color": [184,112,96],
        "synonyms": ["Globules / Clods : Rim of brown globules"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Shiny white blotches and strands",
        "abbreviation": "G/C.:ShWhtB.",
        "color": [192,136,112],
        "synonyms": [
            "Globules / Clods : Shiny white blotches and strands",
            "Shiny white structures : Shiny white blotches and strands"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Spoke wheel areas",
        "abbreviation": "G/C.:SpkWhlA",
        "synonyms": [
            "Globules / Clods : Spoke wheel areas",
            "Lines : Spoke wheel areas",
            "Miscellaneous : Spoke wheel area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : Variant of spoke wheel area",
        "abbreviation": "G/C.:VarSpkW",
        "synonyms": ["Globules / Clods : Variant of spoke wheel area"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Globules / Clods : White",
        "abbreviation": "G/C.:White",
        "synonyms": ["Globules / Clods : White"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : NOS",
        "abbreviation": "Line:NOS",
        "color": [40,40,64],
        "synonyms": ["Lines : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Angulated lines / Polygons / Zig-zag pattern",
        "abbreviation": "Line:Ang/P/Z",
        "color": [96,32,32],
        "synonyms": [
            "Lines : Angulated lines / Polygons / Zig-zag pattern",
            "Miscellaneous : Angulated lines / Polygons"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Atypical pigment network / Reticulation",
        "abbreviation": "Line:AtypPNR",
        "synonyms": [
            "Lines : Atypical pigment network / Reticulation",
            "Network : Atypical pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Branched streaks",
        "abbreviation": "Line:BranSt.",
        "synonyms": ["Lines : Branched streaks"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Broadened pigment network / Reticulation",
        "abbreviation": "Line:Brd.PNR",
        "synonyms": [
            "Lines : Broadened pigment network / Reticulation",
            "Network : Broadened pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Cerebriform pattern",
        "abbreviation": "Line:CerebFP",
        "synonyms": [
            "Lines : Cerebriform pattern",
            "Pattern : Cerebriform"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Crypts",
        "abbreviation": "Line:Crypts",
        "synonyms": [
            "Lines : Crypts",
            "Miscellaneous : Crypts"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Delicate pigment network / Reticulation",
        "abbreviation": "Line:DeliPNR",
        "synonyms": [
            "Lines : Delicate pigment network / Reticulation",
            "Network : Delicate Pigment Network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Fibrillar pattern",
        "abbreviation": "Line:FibrilP",
        "synonyms": [
            "Lines : Fibrillar pattern",
            "Pattern : Fibrilar",
            "Volar lesions : Fibrilar pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Fingerprint pattern",
        "abbreviation": "Line:FingP.P",
        "synonyms": [
            "Lines : Fingerprint pattern",
            "Pattern : Fingerprint"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Leaf like areas",
        "abbreviation": "Line:LeaflkA",
        "synonyms": [
            "Globules / Clods : Leaflike area",
            "Lines : Leaf like areas",
            "Miscellaneous : Leaflike area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Negative pigment network",
        "abbreviation": "Line:Neg.PN.",
        "synonyms": [
            "Lines : Negative pigment network",
            "Network : Negative pigment network"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Parallel furrows pattern (volar lesions)",
        "abbreviation": "Line:ParFurP",
        "synonyms": [
            "Lines : Parallel furrows pattern (volar lesions)",
            "Pattern : Parallel furrow",
            "Volar lesions : Parallel furrow pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Parallel lines (volar lesions)",
        "abbreviation": "Line:ParLine",
        "synonyms": [
            "Lines : Parallel lines (volar lesions)",
            "Pattern : Parallel ridge",
            "Volar lesions : Parallel lines"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Parallel ridge pattern (volar lesions)",
        "abbreviation": "Line:ParRidP",
        "synonyms": [
            "Lines : Parallel ridge pattern (volar lesions)",
            "Volar lesions : Parallel ridge pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Pseudopods",
        "abbreviation": "Line:PseuPd.",
        "synonyms": [
            "Lines : Pseudopods",
            "Miscellaneous : Pseudopods"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Radial streaming",
        "abbreviation": "Line:RadStr.",
        "synonyms": [
            "Lines : Radial streaming",
            "Miscellaneous : Radial streaming"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Rhomboids (facial skin)",
        "abbreviation": "Line:RhbFac.",
        "synonyms": [
            "Facial Skin : Rhomboids / Zig-zag pattern",
            "Lines : Rhomboids (facial skin)",
            "Miscellaneous : Rhomboids / Zig-zag pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Shiny white streaks",
        "abbreviation": "Line:ShWhtS.",
        "synonyms": [
            "Lines : Shiny white streaks",
            "Shiny white structures : Shiny white streaks"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Spoke wheel areas",
        "abbreviation": "Line:SpkWhlA",
        "synonyms": [
            "Globules / Clods : Spoke wheel areas",
            "Lines : Spoke wheel areas",
            "Miscellaneous : Spoke wheel area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Starburst pattern",
        "abbreviation": "Line:StarbP.",
        "synonyms": [
            "Lines : Starburst pattern",
            "Pattern : Starburst"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Streaks",
        "abbreviation": "Line:Streaks",
        "synonyms": [
            "Lines : Streaks",
            "Miscellaneous : Streaks"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Lines : Typical pigment network / Reticulation",
        "abbreviation": "Line:Typ.PNR",
        "synonyms": [
            "Lines : Typical pigment network / Reticulation",
            "Network : Typical pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Angulated lines / Polygons",
        "abbreviation": "Misc:AngLine",
        "synonyms": [
            "Lines : Angulated lines / Polygons / Zig-zag pattern",
            "Miscellaneous : Angulated lines / Polygons"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Annular-granular pattern",
        "abbreviation": "Misc:An-gran",
        "synonyms": [
            "Dots : Annular-granular pattern",
            "Facial Skin : Annular-granular pattern",
            "Miscellaneous : Annular-granular pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Asymmetric pigmented follicular openings",
        "abbreviation": "Misc:Asym.O.",
        "synonyms": ["Miscellaneous : Asymmetric pigmented follicular openings"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Blood spots",
        "abbreviation": "Misc:BloodSp",
        "synonyms": [
            "Miscellaneous : Blood spots",
            "Nail lesions : Blood spots"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Blotch irregular",
        "abbreviation": "Misc:BlotIrr",
        "synonyms": [
            "Miscellaneous : Blotch irregular",
            "Structureless : Blotch irregular"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Blotch regular",
        "abbreviation": "Misc:BlotReg",
        "synonyms": [
            "Miscellaneous : Blotch regular",
            "Structureless : Blotch regular"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Blue-whitish veil",
        "abbreviation": "Misc:Blu-WhV",
        "synonyms": [
            "Miscellaneous : Blue-whitish veil",
            "Structureless : Blue-whitish veil"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Circle within a circle",
        "abbreviation": "Misc:C.w/iC.",
        "synonyms": [
            "Circles & Semicircles : Circle within a circle",
            "Facial Skin : Circle within a circle",
            "Miscellaneous : Circle within a circle"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Comedo-like openings",
        "abbreviation": "Misc:ComedoO",
        "synonyms": [
            "Globules / Clods : Comedo-like openings",
            "Miscellaneous : Comedo-like openings"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Crypts",
        "abbreviation": "Misc:Crypts",
        "synonyms": [
            "Lines : Crypts",
            "Miscellaneous : Crypts"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Disruption of parallelism",
        "abbreviation": "Misc:Disrup.",
        "synonyms": ["Miscellaneous : Disruption of parallelism"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Fissures",
        "abbreviation": "Misc:Fissure",
        "synonyms": ["Miscellaneous : Fissures"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Granularity or granules",
        "abbreviation": "Misc:Granul.",
        "synonyms": ["Miscellaneous : Granularity or granules"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Hutchinson sign",
        "abbreviation": "Misc:HutchS.",
        "synonyms": [
            "Miscellaneous : Hutchinson sign",
            "Nail lesions : Hutchinson sign"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Lacunae : NOS",
        "abbreviation": "Misc:LacuNOS",
        "synonyms": [
            "Globules / Clods : Lacunae : NOS",
            "Miscellaneous : Lacunae : NOS"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Lacunae : Black",
        "abbreviation": "Misc:LacuBlk",
        "synonyms": [
            "Globules / Clods : Lacunae : Black",
            "Miscellaneous : Lacunae : Black"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Lacunae : Blue",
        "abbreviation": "Misc:LacuBlu",
        "synonyms": [
            "Globules / Clods : Lacunae : Blue",
            "Miscellaneous : Lacunae : Blue"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Lacunae : Red",
        "abbreviation": "Misc:LacuRed",
        "color": [160,40,0],
        "synonyms": [
            "Globules / Clods : Lacunae : Red",
            "Miscellaneous : Lacunae : Red"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Lacunae : Red-purple",
        "abbreviation": "Misc:LacuR-P",
        "synonyms": [
            "Globules / Clods : Lacunae : Red-purple",
            "Miscellaneous : Lacunae : Red-purple"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Leaflike area",
        "abbreviation": "Misc:LacuLfA",
        "synonyms": [
            "Globules / Clods : Leaflike area",
            "Lines : Leaf like areas",
            "Miscellaneous : Leaflike area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Milia-like cysts, cloudy or starry",
        "abbreviation": "Misc:MiliaC.",
        "synonyms": [
            "Dots : Milia-like cysts",
            "Globules / Clods : Milia-like cysts",
            "Miscellaneous : Milia-like cysts, cloudy or starry"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Moth-eaten border",
        "abbreviation": "Misc:MothBor",
        "synonyms": ["Miscellaneous : Moth-eaten border"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Pseudo-Hutchinson sign",
        "abbreviation": "Misc:PseuHut",
        "synonyms": [
            "Miscellaneous : Pseudo-Hutchinson sign",
            "Nail lesions : Pseudo-Hutchinson sign"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Pseudonetwork",
        "abbreviation": "Misc:PseuNet",
        "synonyms": [
            "Facial Skin : Pseudonetwork",
            "Miscellaneous : Pseudonetwork",
            "Structureless : Pseudonetwork"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Pseudopods",
        "abbreviation": "Misc:PseuPod",
        "synonyms": [
            "Lines : Pseudopods",
            "Miscellaneous : Pseudopods"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Radial streaming",
        "abbreviation": "Misc:RadStr.",
        "synonyms": [
            "Lines : Radial streaming",
            "Miscellaneous : Radial streaming"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Regularly bended ribbon sign",
        "abbreviation": "Misc:RegBRib",
        "synonyms": ["Miscellaneous : Regularly bended ribbon sign"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Rhomboids / Zig-zag pattern",
        "abbreviation": "Misc:Rhb/Zig",
        "synonyms": [
            "Facial Skin : Rhomboids / Zig-zag pattern",
            "Lines : Rhomboids (facial skin)",
            "Miscellaneous : Rhomboids / Zig-zag pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Ridges",
        "abbreviation": "Misc:Ridges",
        "synonyms": ["Miscellaneous : Ridges"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Scale",
        "abbreviation": "Misc:Scale",
        "synonyms": ["Miscellaneous : Scale"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Setting-sun pattern",
        "abbreviation": "Misc:SunsetP",
        "synonyms": ["Miscellaneous : Setting-sun pattern"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Spoke wheel area",
        "abbreviation": "Misc:SpkWhlA",
        "synonyms": [
            "Globules / Clods : Spoke wheel areas",
            "Lines : Spoke wheel areas",
            "Miscellaneous : Spoke wheel area"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Streaks",
        "abbreviation": "Misc:Streaks",
        "synonyms": [
            "Lines : Streaks",
            "Miscellaneous : Streaks"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : String of pearls",
        "abbreviation": "Misc:SoPearl",
        "synonyms": [
            "Miscellaneous : String of pearls",
            "Vessels : String of pearls"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Structureless, brown (tan)",
        "abbreviation": "Misc:S/lBrwn",
        "synonyms": [
            "Miscellaneous : Structureless, brown (tan)",
            "Structureless : Structureless brown (tan)"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Twisted red loops",
        "abbreviation": "Misc:TwistRL",
        "synonyms": ["Miscellaneous : Twisted red loops"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Miscellaneous : Ulceration / Erosion",
        "abbreviation": "Misc:Ulcer/E",
        "synonyms": ["Miscellaneous : Ulceration / Erosion"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Blood spots",
        "abbreviation": "Nail:BloodSp",
        "synonyms": [
            "Miscellaneous : Blood spots",
            "Nail lesions : Blood spots"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Granular inclusions",
        "abbreviation": "Nail:GranInc",
        "synonyms": ["Nail lesions : Granular inclusions"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Hutchinson sign",
        "abbreviation": "Nail:HutchS.",
        "synonyms": [
            "Miscellaneous : Hutchinson sign",
            "Nail lesions : Hutchinson sign"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Irregular lines",
        "abbreviation": "Nail:Irr.Lin",
        "synonyms": ["Nail lesions : Irregular lines"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Pseudo-Hutchinson sign",
        "abbreviation": "Nail:PseuHut",
        "synonyms": [
            "Miscellaneous : Pseudo-Hutchinson sign",
            "Nail lesions : Pseudo-Hutchinson sign"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Regular lines",
        "abbreviation": "Nail:Reg.Lin",
        "synonyms": ["Nail lesions : Regular lines"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Nail lesions : Splinter hemorrhage",
        "abbreviation": "Nail:SplintH",
        "synonyms": ["Nail lesions : Splinter hemorrhage"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : NOS",
        "abbreviation": "Net:NOS",
        "synonyms": ["Network : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : Atypical pigment network / Reticulation",
        "abbreviation": "Net:AtypPNR",
        "synonyms": [
            "Lines : Atypical pigment network / Reticulation",
            "Network : Atypical pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : Broadened pigment network / Reticulation",
        "abbreviation": "Net:Brd.PNR",
        "synonyms": [
            "Lines : Broadened pigment network / Reticulation",
            "Network : Broadened pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : Delicate Pigment Network / Reticulation",
        "abbreviation": "Net:DeliPNR",
        "synonyms": [
            "Lines : Delicate pigment network / Reticulation",
            "Network : Delicate Pigment Network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : Negative pigment network",
        "abbreviation": "Net:Neg.PN.",
        "synonyms": [
            "Lines : Negative pigment network",
            "Network : Negative pigment network"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Network : Typical pigment network / Reticulation",
        "abbreviation": "Net:Typ.PNR",
        "synonyms": [
            "Lines : Typical pigment network / Reticulation",
            "Network : Typical pigment network / Reticulation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : NOS",
        "abbreviation": "Patt:NOS",
        "synonyms": ["Pattern : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Cerebriform",
        "abbreviation": "Patt:CerebF.",
        "synonyms": [
            "Lines : Cerebriform pattern",
            "Pattern : Cerebriform"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Fibrilar",
        "abbreviation": "Patt:Fibril.",
        "synonyms": [
            "Lines : Fibrillar pattern",
            "Pattern : Fibrilar",
            "Volar lesions : Fibrilar pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Fingerprint",
        "abbreviation": "Patt:FingerP",
        "synonyms": [
            "Lines : Fingerprint pattern",
            "Pattern : Fingerprint"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Homogeneous : NOS",
        "abbreviation": "Patt:Hom.NOS",
        "synonyms": [
            "Pattern : Homogeneous : NOS",
            "Structureless : Homogeneous pattern : NOS"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Homogeneous : Blue",
        "abbreviation": "Patt:Hom.Blu",
        "synonyms": [
            "Pattern : Homogeneous : Blue",
            "Structureless : Homogeneous pattern : Blue"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Homogeneous : Brown",
        "abbreviation": "Patt:Hom.Brw",
        "synonyms": [
            "Pattern : Homogeneous : Brown",
            "Structureless : Homogeneous pattern : Brown"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Homogeneous : Pink",
        "abbreviation": "Patt:Hom.Pnk",
        "synonyms": [
            "Pattern : Homogeneous : Pink",
            "Structureless : Homogeneous pattern : Pink"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Honeycomb",
        "abbreviation": "Patt:Honeycb",
        "synonyms": ["Pattern : Honeycomb"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Latticelike",
        "abbreviation": "Patt:Lattice",
        "synonyms": [
            "Pattern : Latticelike",
            "Volar lesions : Latticelike pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Parallel furrow",
        "abbreviation": "Patt:Par.Fur",
        "synonyms": [
            "Lines : Parallel furrows pattern (volar lesions)",
            "Pattern : Parallel furrow",
            "Volar lesions : Parallel furrow pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Parallel ridge",
        "abbreviation": "Patt:Par.Rid",
        "synonyms": [
            "Lines : Parallel lines (volar lesions)",
            "Pattern : Parallel ridge",
            "Volar lesions : Parallel lines"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Rainbow",
        "abbreviation": "Patt:Rainbow",
        "synonyms": [
            "Pattern : Rainbow",
            "Structureless : Rainbow pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Starburst",
        "abbreviation": "Patt:Starb.",
        "synonyms": [
            "Lines : Starburst pattern",
            "Pattern : Starburst"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Pattern : Strawberry",
        "abbreviation": "Patt:Strawb.",
        "synonyms": [
            "Pattern : Strawberry",
            "Structureless : Strawberry pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Regression structures : NOS",
        "abbreviation": "RegS:NOS",
        "synonyms": ["Regression structures : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Regression structures : Peppering / Granularity",
        "abbreviation": "RegS:PepperG",
        "synonyms": [
            "Dots : Peppering",
            "Regression structures : Peppering / Granularity"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Regression structures : Scarlike depigmentation",
        "abbreviation": "RegS:ScarlkD",
        "synonyms": [
            "Regression structures : Scarlike depigmentation",
            "Structureless : Scar-like depigmentation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Shiny white structures : NOS",
        "abbreviation": "SWhS:NOS",
        "synonyms": ["Shiny white structures : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Shiny white structures : Rosettes",
        "abbreviation": "SWhS:Rosette",
        "synonyms": [
            "Dots : Rosettes",
            "Shiny white structures : Rosettes"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Shiny white structures : Shiny white blotches and strands",
        "abbreviation": "SWhS:ShWhtB.",
        "synonyms": [
            "Globules / Clods : Shiny white blotches and strands",
            "Shiny white structures : Shiny white blotches and strands"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Shiny white structures : Shiny white streaks",
        "abbreviation": "SWhS:ShWhtS.",
        "synonyms": [
            "Lines : Shiny white streaks",
            "Shiny white structures : Shiny white streaks"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : NOS",
        "abbreviation": "SLes:NOS",
        "synonyms": ["Structureless : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Blotch irregular",
        "abbreviation": "SLes:BlotIrr",
        "synonyms": [
            "Miscellaneous : Blotch irregular",
            "Structureless : Blotch irregular"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Blotch regular",
        "abbreviation": "SLes:BlotReg",
        "synonyms": [
            "Miscellaneous : Blotch regular",
            "Structureless : Blotch regular"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Blue-whitish veil",
        "abbreviation": "SLes:Blu-WhV",
        "synonyms": [
            "Miscellaneous : Blue-whitish veil",
            "Structureless : Blue-whitish veil"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Central white patch",
        "abbreviation": "SLes:CentWhP",
        "synonyms": ["Structureless : Central white patch"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Homogeneous pattern : NOS",
        "abbreviation": "SLes:Hom.NOS",
        "synonyms": [
            "Pattern : Homogeneous : NOS",
            "Structureless : Homogeneous pattern : NOS"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Homogeneous pattern : Blue",
        "abbreviation": "SLes:Hom.Blu",
        "synonyms": [
            "Pattern : Homogeneous : Blue",
            "Structureless : Homogeneous pattern : Blue"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Homogeneous pattern : Brown",
        "abbreviation": "SLes:Hom.Brw",
        "synonyms": [
            "Pattern : Homogeneous : Brown",
            "Structureless : Homogeneous pattern : Brown"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Homogeneous pattern : Pink",
        "abbreviation": "SLes:Hom.Pnk",
        "synonyms": [
            "Pattern : Homogeneous : Pink",
            "Structureless : Homogeneous pattern : Pink"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Milky red areas",
        "abbreviation": "SLes:MilkRed",
        "synonyms": ["Structureless : Milky red areas"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Pseudonetwork",
        "abbreviation": "SLes:PseuNet",
        "synonyms": [
            "Facial Skin : Pseudonetwork",
            "Miscellaneous : Pseudonetwork",
            "Structureless : Pseudonetwork"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Rainbow pattern",
        "abbreviation": "SLes:Rainbow",
        "synonyms": [
            "Pattern : Rainbow",
            "Structureless : Rainbow pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Scar-like depigmentation",
        "abbreviation": "SLes:ScarlkD",
        "synonyms": [
            "Regression structures : Scarlike depigmentation",
            "Structureless : Scar-like depigmentation"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Strawberry pattern",
        "abbreviation": "SLes:Strawb.",
        "synonyms": [
            "Pattern : Strawberry",
            "Structureless : Strawberry pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Structureless : Structureless brown (tan)",
        "abbreviation": "SLes:Brown/T",
        "synonyms": [
            "Miscellaneous : Structureless, brown (tan)",
            "Structureless : Structureless brown (tan)"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : NOS",
        "abbreviation": "Vess:NOS",
        "synonyms": ["Vessels : NOS"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Arborizing",
        "abbreviation": "Vess:Arbor.",
        "color": [128,128,32],
        "synonyms": ["Vessels : Arborizing"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Comma",
        "abbreviation": "Vess:Comma",
        "synonyms": ["Vessels : Comma"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Corkscrew",
        "abbreviation": "Vess:Corksc.",
        "synonyms": ["Vessels : Corkscrew"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Crown",
        "abbreviation": "Vess:Crown",
        "synonyms": ["Vessels : Crown"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Dotted",
        "abbreviation": "Vess:Dotted",
        "color": [192,32,32],
        "synonyms": ["Vessels : Dotted"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Glomerular",
        "abbreviation": "Vess:Glomer.",
        "synonyms": ["Vessels : Glomerular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Hairpin",
        "abbreviation": "Vess:Hairpin",
        "synonyms": ["Vessels : Hairpin"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Linear irregular",
        "abbreviation": "Vess:Lin.Irr",
        "synonyms": ["Vessels : Linear irregular"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Monomorphous",
        "abbreviation": "Vess:Monomrp",
        "synonyms": ["Vessels : Monomorphous"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Polymorphous",
        "abbreviation": "Vess:Polymrp",
        "synonyms": ["Vessels : Polymorphous"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : String of pearls",
        "abbreviation": "Vess:SoPearl",
        "synonyms": [
            "Miscellaneous : String of pearls",
            "Vessels : String of pearls"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Vessels : Targetoid",
        "abbreviation": "Vess:Target.",
        "synonyms": ["Vessels : Targetoid"]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Volar lesions : Fibrilar pattern",
        "abbreviation": "VolL:FibrilP",
        "synonyms": [
            "Lines : Fibrillar pattern",
            "Pattern : Fibrilar",
            "Volar lesions : Fibrilar pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Volar lesions : Latticelike pattern",
        "abbreviation": "VolL:Lattice",
        "synonyms": [
            "Pattern : Latticelike",
            "Volar lesions : Latticelike pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Volar lesions : Parallel furrow pattern",
        "abbreviation": "VolL:Par.Fur",
        "synonyms": [
            "Lines : Parallel furrows pattern (volar lesions)",
            "Pattern : Parallel furrow",
            "Volar lesions : Parallel furrow pattern"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Volar lesions : Parallel lines",
        "abbreviation": "VolL:Par.Lin",
        "synonyms": [
            "Lines : Parallel lines (volar lesions)",
            "Pattern : Parallel ridge",
            "Volar lesions : Parallel lines"
        ]
    },
    {
        "nomenclature": "metaphoric",
        "id": "Volar lesions : Parallel ridge pattern",
        "abbreviation": "VolL:Par.Rid",
        "synonyms": [
            "Lines : Parallel ridge pattern (volar lesions)",
            "Volar lesions : Parallel ridge pattern"
        ]
    }
]

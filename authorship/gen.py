import pandas as pd

kjv = pd.read_csv("source_data/t_kjv.csv")

kjv.rename(
    columns={
        "field": "id",
        "field.1": "book",
        "field.2": "chapter",
        "field.3": "verse",
        "field.4": "text",
    },
    inplace=True,
)

key_df = pd.read_csv("source_data/key_english.csv")

books_key = pd.Series(key_df["field.1"].values, index=key_df["field"]).to_dict()
kjv["book"] = kjv["book"].map(books_key)
kjv["book"] = kjv["book"].convert_dtypes()
kjv["text"] = kjv["text"].convert_dtypes()

# Training datasets
undisputed_paul = kjv.loc[
    (kjv["book"] == "Romans")
    | (kjv["book"] == "1 Corinthians")
    | (kjv["book"] == "2 Corinthians")
    | (kjv["book"] == "Galatians")
    | (kjv["book"] == "Philippians")
    | (kjv["book"] == "1 Thessalonians")
    | (kjv["book"] == "Philemon")
]

undisputed_paul.to_csv("undisputed_paul.csv", index=False)

undisputed_catholic = kjv.loc[
    (kjv["book"] == "James")
    | (kjv["book"] == "1 Peter")
    | (kjv["book"] == "2 Peter")
    | (kjv["book"] == "1 John")
    | (kjv["book"] == "2 John")
    | (kjv["book"] == "3 John")
    | (kjv["book"] == "Jude")
]

undisputed_catholic.to_csv("undisputed_catholic.csv", index=False)

# Target datasets
disputed_deuteropauline = kjv.loc[
    (kjv["book"] == "Ephesians")
    | (kjv["book"] == "Colossians")
    | (kjv["book"] == "2 Thessalonians")
]

disputed_deuteropauline.to_csv("disputed_deuteropauline.csv", index=False)


disputed_pastoral = kjv.loc[
    (kjv["book"] == "1 Timothy")
    | (kjv["book"] == "2 Timothy")
    | (kjv["book"] == "Titus")
]

disputed_pastoral.to_csv("disputed_pastoral.csv", index=False)


disputed_sermon = kjv.loc[(kjv["book"] == "Hebrews")]

disputed_sermon.to_csv("disputed_sermon.csv", index=False)


# Create training datasets
undisputed_paul_sample_1 = undisputed_paul.sample(n=250, random_state=1)
undisputed_paul_sample_2 = undisputed_paul.sample(n=250, random_state=2)
undisputed_paul_sample_3 = undisputed_paul.sample(n=250, random_state=3)

undisputed_paul_sample_1["authentic"] = True
undisputed_paul_sample_2["authentic"] = True
undisputed_paul_sample_3["authentic"] = True

undisputed_paul_sample_1.to_csv(
    "training_data/undisputed_paul_sample_1.csv", index=False
)
undisputed_paul_sample_2.to_csv(
    "training_data/undisputed_paul_sample_2.csv", index=False
)
undisputed_paul_sample_3.to_csv(
    "training_data/undisputed_paul_sample_3.csv", index=False
)

undisputed_catholic_sample_1 = undisputed_catholic.sample(n=250, random_state=1)
undisputed_catholic_sample_2 = undisputed_catholic.sample(n=250, random_state=2)
undisputed_catholic_sample_3 = undisputed_catholic.sample(n=250, random_state=3)

undisputed_catholic_sample_1["authentic"] = False
undisputed_catholic_sample_2["authentic"] = False
undisputed_catholic_sample_3["authentic"] = False

# undisputed_catholic_sample_1.to_csv(
#     "training_data/undisputed_catholic_sample_1.csv", index=False
# )
# undisputed_catholic_sample_2.to_csv(
#     "training_data/undisputed_catholic_sample_2.csv", index=False
# )
# undisputed_catholic_sample_3.to_csv(
#     "training_data/undisputed_catholic_sample_3.csv", index=False
# )

undisputed_paul_ids = pd.concat(
    [
        undisputed_paul_sample_1["id"],
        undisputed_paul_sample_2["id"],
        undisputed_paul_sample_3["id"],
    ],
    ignore_index=True,
)

undisputed_catholic_ids = pd.concat(
    [
        undisputed_catholic_sample_1["id"],
        undisputed_catholic_sample_2["id"],
        undisputed_catholic_sample_3["id"],
    ],
    ignore_index=True,
)

validation_paul = undisputed_paul[~undisputed_paul["id"].isin(undisputed_paul_ids)]
validation_catholic = undisputed_catholic[
    ~undisputed_catholic["id"].isin(undisputed_catholic_ids)
]

print("paul: ", undisputed_paul.size, validation_paul.size)
print("catholic: ", undisputed_catholic.size, validation_catholic.size)

undisputed_catholic_ids.to_csv("cath_ids.csv", index=False)
undisputed_catholic.to_csv("undisp_cath.csv", index=False)

validation_paul.to_csv(
    "validation_data/validation_paul.csv", index=False, line_terminator="\n"
)
validation_catholic.to_csv(
    "validation_data/validation_catholic.csv", index=False, line_terminator="\n"
)

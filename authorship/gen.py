import pandas as pd

kjv = pd.read_csv("0_source_data/t_kjv.csv")
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

key_df = pd.read_csv("0_source_data/key_english.csv")

books_key = pd.Series(key_df["field.1"].values, index=key_df["field"]).to_dict()

kjv["book"] = kjv["book"].map(books_key)
kjv["book"] = kjv["book"].convert_dtypes()
kjv["text"] = kjv["text"].convert_dtypes()

# Create training dataset - Authentic
undisputed_paul = kjv.loc[
    (kjv["book"] == "Romans")
    | (kjv["book"] == "1 Corinthians")
    | (kjv["book"] == "2 Corinthians")
    | (kjv["book"] == "Galatians")
    | (kjv["book"] == "Philippians")
    | (kjv["book"] == "1 Thessalonians")
    | (kjv["book"] == "Philemon")
]

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

print("Created dataset - Authentic")

# Create training dataset - Not Authentic
undisputed_catholic = kjv.loc[
    (kjv["book"] == "James")
    | (kjv["book"] == "1 Peter")
    | (kjv["book"] == "2 Peter")
    | (kjv["book"] == "1 John")
    | (kjv["book"] == "2 John")
    | (kjv["book"] == "3 John")
    | (kjv["book"] == "Jude")
]

undisputed_catholic_sample_1 = undisputed_catholic.sample(n=250, random_state=1)
undisputed_catholic_sample_2 = undisputed_catholic.sample(n=250, random_state=2)
undisputed_catholic_sample_3 = undisputed_catholic.sample(n=250, random_state=3)

undisputed_catholic_sample_1["authentic"] = False
undisputed_catholic_sample_2["authentic"] = False
undisputed_catholic_sample_3["authentic"] = False

undisputed_catholic_sample_1.to_csv(
    "training_data/undisputed_catholic_sample_1.csv", index=False
)
undisputed_catholic_sample_2.to_csv(
    "training_data/undisputed_catholic_sample_2.csv", index=False
)
undisputed_catholic_sample_3.to_csv(
    "training_data/undisputed_catholic_sample_3.csv", index=False
)

print("Created datasets - Not Authentic")

# Create validation sets
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

validation_paul.to_csv("validation_data/validation_paul", index=False)
validation_catholic.to_csv("validation_data/validation_catholic", index=False)

print("Created datasets - Validation")

# Create target datasets
disputed_deuteropauline = kjv.loc[
    (kjv["book"] == "Ephesians")
    | (kjv["book"] == "Colossians")
    | (kjv["book"] == "2 Thessalonians")
]

disputed_pastoral = kjv.loc[
    (kjv["book"] == "1 Timothy")
    | (kjv["book"] == "2 Timothy")
    | (kjv["book"] == "Titus")
]

disputed_sermon = kjv.loc[(kjv["book"] == "Hebrews")]

print("Created dataset - Target")

### Analyze validation set
validated_p = pd.read_csv("4_validation_results/validation_paul.csv")
validated_c = pd.read_csv("4_validation_results/validation_catholic.csv")

mean_p = validated_p["authentic"].mean()
mean_c = validated_c["authentic"].mean()


median_p = validated_p["authentic"].median()
median_c = validated_c["authentic"].median()

std_p = validated_p["authentic"].std()
std_c = validated_c["authentic"].std()

print("")
print(
    "                             Mean                   Median                 Standard Dev."
)
print(
    "----------------------------------------------------------------------------------------------"
)
print(f"Validated Paul         {mean_p}      {median_p}        {std_p}")
print(f"Validated Not Paul     {mean_c}      {median_c}      {std_c}")
print("")

import matplotlib.pyplot as plt
import scipy
import seaborn as sns

fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
labels = ["Paul", "Not Paul"]
ax0.hist(
    [validated_p["authentic"], validated_c["authentic"]],
    10,
    histtype="bar",
    density=1,
    label=labels,
)
ax0.legend(prop={"size": 10})
ax0.set_title("Validation set")
ax0.set_xlabel("Authenticity score")
ax0.set_ylabel("Count (prob. density)")
plt.show()

### Analyze results
result_dp = pd.read_csv("6_results/disputed_deuteropauline.csv")
result_p = pd.read_csv("6_results/disputed_pastoral.csv")
result_s = pd.read_csv("6_results/disputed_sermon.csv")

mean_dp = result_dp["authentic"].mean()
mean_p = result_p["authentic"].mean()
mean_s = result_s["authentic"].mean()

median_dp = result_dp["authentic"].median()
median_p = result_p["authentic"].median()
median_s = result_s["authentic"].median()

std_dp = result_dp["authentic"].std()
std_p = result_p["authentic"].std()
std_s = result_s["authentic"].std()


print("")
print(
    "                           Mean                   Median                 Standard Dev."
)
print(
    "----------------------------------------------------------------------------------------------"
)
print(f"Deutero-Pauline     {mean_dp}       {median_dp}         {std_dp}")
print(f"Pastoral            {mean_p}      {median_p}       {std_p}")
print(f"Hebrews             {mean_s}       {median_s}        {std_s} ")
print("")

fig, ((ax0)) = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
labels = ["Deutero-Pauline", "Pastoral", "Sermon"]
ax0.hist(
    [result_dp["authentic"], result_p["authentic"], result_s["authentic"]],
    10,
    histtype="bar",
    label=labels,
)
ax0.legend(prop={"size": 10})
ax0.set_title("Disputed epistles")
ax0.set_xlabel("Authenticity score")
ax0.set_ylabel("Count")
plt.show()

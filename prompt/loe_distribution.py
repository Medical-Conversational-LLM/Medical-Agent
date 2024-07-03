import pandas as pd
import matplotlib.pyplot as plt
def distribution():

    df = pd.read_csv('../storage/datasets/updated_PubMedQA_pqa_artificial.csv')

    #Check unique values in the 'LoE' column
    loe_values = df['loe'].unique()
    print("Unique LoE values:", loe_values)

    #Calculate distribution of 'LoE' values
    # loe_distribution = df['loe'].value_counts()
    loe_distribution = {3: 69582, 6: 63179, 4: 29612, 1: 23304, 0: 11073, 2: 9599, 5: 2277}

    # Extract keys and values
    loe_dict =  {'1a': 11073, '1b': 23304, '2a': 9599, '2b': 69582, '3a': 29612, '3b': 2277, '4': 63179}

    categories = list(loe_dict.keys())
    counts = list(loe_dict.values())

    print("\nLoE Distribution:")
    print(loe_dict)



    colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'yellow', 'cyan']


    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color=colors)
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Distribution of Levels of Evidence (LoE)')
    plt.xticks(categories)  # Ensure all categories are shown on x-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()


    plt.savefig('loe_dist.png')
    plt.show()



distribution()
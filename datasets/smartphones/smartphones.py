import pandas as pd

file_path = 'scratch\\datasets\\smartphones\\smartphones.csv'
df = pd.read_csv(file_path)

# most used color
color_count = df['Color'].value_counts()

most_used_color = color_count.idxmax()
most_used_count = color_count.max()

# most common brand based on color
brand_count = df.groupby('Color')['Brand'].value_counts()

most_used_brand = brand_count.idxmax()
most_used_brand_count = brand_count.max()


print(f"Most used brand: {most_used_brand} ({most_used_brand_count} times)")


print(f"Most used color: {most_used_color} ({most_used_count} times)")

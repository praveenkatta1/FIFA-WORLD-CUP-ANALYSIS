import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set Seaborn style
sns.set_style('whitegrid')
import warnings 
warnings.filterwarnings('ignore')

# Title and introduction
st.title('FIFA WORLD CUP 2019 EXTENSIVE Data Analysis By KATTA SRI SAI PRAVEEN')
st.write('This is a simple web app to analyze the FIFA World Cup 2019 data (assumed FIFA 19 game data).')

# Upload FIFA image
st.subheader('Upload FIFA 2019 Image')
fifa_image = Image.open(r"C:\Users\srisa\Downloads\FIFA 2019.jpg")
st.image(fifa_image, caption='FIFA 2019')

# Upload dataset
st.subheader('IMPORT THE DATA SET')
fifa = pd.read_csv(r'C:\Users\srisa\Desktop\FIFA.csv')
st.write(fifa.head())

# Display column names
st.subheader('COLUMN NAMES')
st.write(fifa.columns)

# Dataset summary
st.subheader('VIEW THE SUMMARY OF THE DATA SET')
st.write(fifa.describe())

# Dataset description using Markdown
st.markdown("""
- This dataset contains 89 variables.
- Out of the 89 variables, 44 are numerical variables: 38 are float64, 6 are int64.
- The remaining 45 variables are of character data type.
""")

# Deleting a column
st.subheader("DELETING A COLUMN")
fifa = fifa.drop("Unnamed: 0", axis=1)
st.write(fifa.head())

# Additional Analysis
st.subheader("Body Type Distribution")
st.write(fifa["Body Type"].value_counts())

# Explore Age variable
st.subheader("Explore `Age` variable")
st.subheader("Visualize distribution of `Age` with Seaborn `histplot()`")
st.markdown("""
- Seaborn `histplot()` plots a univariate distribution of observations.
- It combines Matplotlib’s histogram with kernel density estimation (KDE).
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(fifa["Age"], bins=20, color="r", kde=True, ax=ax)
plt.title("Age Distribution")
st.pyplot(fig)
plt.close(fig)
st.markdown("- The `Age` variable is slightly positively skewed.")

fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(pd.Series(fifa['Age'], name="Age variable"), bins=10, color="y", kde=True, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.subheader("- Plotting distribution on the vertical axis")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(y=fifa['Age'], bins=10, color="g", kde=True, ax=ax)
st.pyplot(fig)
plt.close(fig)

# KDE Plot
st.subheader("Seaborn Kernel Density Estimation (KDE) Plot")
st.markdown("""
- The KDE plot visualizes the shape of a distribution.
- Seaborn `kdeplot` fits and plots a univariate or bivariate kernel density estimate.
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.kdeplot(fifa['Age'], color="b", ax=ax)
plt.title("Age Distribution")
st.pyplot(fig)
plt.close(fig)

st.markdown("- Shading under the density curve with a different color:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.kdeplot(fifa['Age'], fill=True, color="g", ax=ax)
plt.title("Age Distribution")
st.pyplot(fig)
plt.close(fig)

# Histograms
st.subheader("Histograms")
st.markdown("""
- A histogram shows the distribution of data by forming bins and drawing bars for observation counts.
- Seaborn enhances Matplotlib’s `hist()` functionality.
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(fifa['Age'], kde=False, bins=10, ax=ax)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(fifa['Age'], color="g", ax=ax)
st.pyplot(fig)
plt.close(fig)

# Explore Preferred Foot variable
st.subheader("Explore `Preferred Foot` variable")
st.subheader("Number of unique values in `Preferred Foot`")
st.write(fifa["Preferred Foot"].nunique())
st.markdown("- There are two unique values in `Preferred Foot`.")

st.subheader("Frequency distribution of `Preferred Foot`")
st.write(fifa["Preferred Foot"].value_counts())
st.markdown("- Values are `Right` and `Left`.")

st.subheader("Visualize with Seaborn `countplot()`")
st.markdown("""
- A `countplot` shows observation counts in categorical bins using bars.
- It’s like a histogram for categorical variables.
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(x="Preferred Foot", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x="Preferred Foot", hue="Real Face", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Vertical countplot:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.countplot(x=fifa["Preferred Foot"], color="r", ax=ax)
plt.title("Preferred Foot Distribution")
st.pyplot(fig)
plt.close(fig)

# Catplot
st.subheader("Seaborn `catplot()`")
st.markdown("""
- `catplot()` draws categorical plots onto a FacetGrid.
- Default is a scatterplot; `kind='count'` makes it a count plot.
""")
c = sns.catplot(x="Preferred Foot", data=fifa, kind="count", palette="ch:.25", height=6, aspect=1.5)
st.pyplot(c.figure)
plt.close(c.figure)

# Explore International Reputation variable
st.subheader("Explore `International Reputation` variable")
st.subheader("Unique values in `International Reputation`")
st.write(fifa["International Reputation"].nunique())

st.subheader("Distribution of `International Reputation`")
st.write(fifa["International Reputation"].value_counts())

st.subheader("Seaborn `stripplot()`")
st.markdown("""
- `stripplot` draws a scatterplot with a categorical variable.
- Useful with box or violin plots to show all observations.
""")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("With jitter:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", data=fifa, jitter=0.01, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Nested with `Preferred Foot`:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, jitter=0.2, palette="Set2", dodge=True, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Large points with different aesthetics:")
fig, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, jitter=0.2, palette="Set1", size=20, marker="D", dodge=True, edgecolor="gray", linewidth=1, alpha=0.25, ax=ax)
plt.title("International Reputation vs Potential")
st.pyplot(fig)
plt.close(fig)

# Boxplot
st.subheader("Seaborn `boxplot()`")
st.markdown("""
- `boxplot` shows distributions with respect to categories using quartiles and whiskers.
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x=fifa["Potential"], ax=ax)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x="International Reputation", y="Potential", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Nested grouping by two variables:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set2", ax=ax)
st.pyplot(fig)
plt.close(fig)

# Violinplot
st.subheader("Seaborn `violinplot()`")
st.markdown("""
- `violinplot` combines boxplot and KDE to show distribution across categorical variables.
""")
fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(x=fifa["Potential"], color="r", ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Grouped by `International Reputation`:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(x="International Reputation", y="Potential", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Nested grouping by two variables:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set1", ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Split violins:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.violinplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set2", split=True, ax=ax)
st.pyplot(fig)
plt.close(fig)

# Pointplot
st.subheader("Seaborn `pointplot()`")
st.markdown("""
- `pointplot` shows point estimates and confidence intervals with scatter glyphs.
""")
fig, ax = plt.subplots(figsize=(7, 6))
sns.pointplot(x="International Reputation", y="Potential", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Nested grouping by two variables:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set1", ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Separated points:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, dodge=True, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("No linking:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, dodge=True, join=False, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Different markers and linestyles:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.pointplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, dodge=True, join=False, markers=["o", "x"], linestyles=["-", "--"], ax=ax)
st.pyplot(fig)
plt.close(fig)

# Barplot
st.subheader("Seaborn `barplot()`")
st.markdown("""
- `barplot` shows point estimates and confidence intervals as bars.
""")
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa, ax=ax)
plt.title("International Reputation vs Potential")
st.pyplot(fig)
plt.close(fig)

st.markdown("Nested grouping by two variables:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set1", ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Using median:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", hue="Preferred Foot", data=fifa, palette="Set2", estimator=np.median, ax=ax)
plt.title("International Reputation vs Potential")
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa, estimator=np.median, ci=68, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("Standard deviation instead of CI:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa, estimator=np.median, ci="sd", ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("With caps on error bars:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.barplot(x="International Reputation", y="Potential", data=fifa, estimator=np.median, ci=95, capsize=0.3, palette="Set1", ax=ax)
st.pyplot(fig)
plt.close(fig)

# Relplot
st.subheader("Visualizing statistical relationship with Seaborn `relplot()`")
st.subheader("Seaborn `relplot()`")
st.markdown("""
- `relplot()` draws relational plots onto a FacetGrid with `kind='scatter'` or `kind='line'`.
""")
g = sns.relplot(x="Overall", y="Potential", data=fifa)
st.pyplot(g.figure)
plt.close(g.figure)

g = sns.relplot(x="Jersey Number", y="Overall", hue="Preferred Foot", data=fifa, palette="Set1")
st.pyplot(g.figure)
plt.close(g.figure)

# Scatterplot
st.subheader("Seaborn `scatterplot()`")
st.markdown("""
- `scatterplot` draws a scatter plot with semantic groupings via `hue`, `size`, and `style`.
""")
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x="Height", y="Weight", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

# Lineplot
st.subheader("Seaborn `lineplot()`")
st.markdown("""
- `lineplot` draws a line plot with semantic groupings.
""")
fig, ax = plt.subplots(figsize=(7, 6))
sns.lineplot(x="Stamina", y="Strength", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.lineplot(x="Stamina", y="Strength", hue="Preferred Foot", data=fifa, palette="Set1", ax=ax)
plt.title("Stamina vs Strength")
st.pyplot(fig)
plt.close(fig)

# Regplot
st.subheader("Visualize linear relationship with Seaborn `regplot()`")
st.markdown("""
- `regplot()` plots data and a linear regression model fit.
""")
fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x="Stamina", y="Strength", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x="Overall", y="Potential", data=fifa, ax=ax)
plt.legend(["Overall vs Potential"])
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x="Overall", y="Potential", data=fifa, fit_reg=False, ax=ax)
plt.legend(["Overall vs Potential"])
st.pyplot(fig)
plt.close(fig)

st.markdown("Different color and marker:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x="Overall", y="Potential", data=fifa, color="y", marker="x", ax=ax)
plt.title("Overall vs Potential")
st.pyplot(fig)
plt.close(fig)

st.markdown("Discrete variable with jitter:")
fig, ax = plt.subplots(figsize=(7, 6))
sns.regplot(x="International Reputation", y="Potential", data=fifa, x_jitter=0.02, ax=ax)
plt.title("International Reputation vs Potential")
st.pyplot(fig)
plt.close(fig)

# Lmplot
st.subheader("Seaborn `lmplot()`")
st.markdown("""
- `lmplot()` combines `regplot()` and `FacetGrid` for regression across subsets.
""")
g = sns.lmplot(x="Overall", y="Potential", data=fifa)
st.pyplot(g.figure)
plt.close(g.figure)

st.markdown("Different color palette:")
g = sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa, palette="Set1")
plt.title("Overall vs Potential")
st.pyplot(g.figure)
plt.close(g.figure)

st.markdown("Condition on third variable:")
g = sns.lmplot(x="Overall", y="Potential", hue="Preferred Foot", data=fifa)
plt.title("Overall vs Potential")
st.pyplot(g.figure)
plt.close(g.figure)

st.markdown("Across different columns:")
g = sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa)
st.pyplot(g.figure)
plt.close(g.figure)

# FacetGrid
st.subheader("MULTI PLOT GRIDS")
st.subheader("Seaborn `FacetGrid()`")
st.markdown("""
- `FacetGrid` visualizes variable distributions within dataset subsets.
""")
f = sns.FacetGrid(fifa, col="Preferred Foot")
st.pyplot(f.figure)
plt.close(f.figure)

st.markdown("Univariate plot of `Overall`:")
f = sns.FacetGrid(fifa, col="Preferred Foot")
f = f.map(plt.hist, "Overall")
st.pyplot(f.figure)
plt.close(f.figure)

st.markdown("Univariate plot of `Potential`:")
f = sns.FacetGrid(fifa, col="Preferred Foot", hue="Preferred Foot")
f = f.map(plt.hist, "Potential")
st.pyplot(f.figure)
plt.close(f.figure)

f = sns.FacetGrid(fifa, col="Preferred Foot", hue="Preferred Foot")
f = f.map(plt.hist, "Potential", bins=10, color="g")
st.pyplot(f.figure)
plt.close(f.figure)

st.markdown("Bivariate plot:")
f = sns.FacetGrid(fifa, col="Preferred Foot", hue="Preferred Foot")
f = f.map(plt.scatter, "Height", "Weight", edgecolor="w").add_legend()
st.pyplot(f.figure)
plt.close(f.figure)

st.markdown("Custom size:")
f = sns.FacetGrid(fifa, col="Preferred Foot", hue="Preferred Foot", height=6, aspect=1.5)
f = f.map(plt.hist, "Potential")
st.pyplot(f.figure)
plt.close(f.figure)

# PairGrid
st.subheader("Seaborn `PairGrid()`")
st.markdown("""
- `PairGrid` plots pairwise relationships in a grid.
""")
fifa_new = fifa[['Age', 'Potential', 'Strength', 'Stamina', 'Preferred Foot']]
p = sns.PairGrid(fifa_new)
p = p.map(plt.scatter)
st.pyplot(p.figure)
plt.close(p.figure)

st.markdown("Univariate distribution on diagonal:")
p = sns.PairGrid(fifa_new)
p = p.map_diag(plt.hist)
p = p.map_offdiag(plt.scatter)
st.pyplot(p.figure)
plt.close(p.figure)

p = sns.PairGrid(fifa_new, hue="Preferred Foot")
p = p.map_diag(plt.hist)
p = p.map_offdiag(plt.scatter)
p = p.add_legend()
st.pyplot(p.figure)
plt.close(p.figure)

st.markdown("Different histogram style:")
p = sns.PairGrid(fifa_new, hue="Preferred Foot", palette="Set1")
p = p.map_diag(plt.hist, histtype="step", linewidth=3)
p = p.map_offdiag(plt.scatter)
p = p.add_legend()
st.pyplot(p.figure)
plt.close(p.figure)

st.markdown("Subset of variables:")
p = sns.PairGrid(fifa_new, vars=["Potential", "Strength"])
p = p.map(plt.scatter)
st.pyplot(p.figure)
plt.close(p.figure)

p = sns.PairGrid(fifa_new, vars=["Age", "Stamina"], hue="Preferred Foot")
p = p.map(plt.scatter)
p = p.add_legend()
st.pyplot(p.figure)
plt.close(p.figure)

st.markdown("Different functions on triangles:")
p = sns.PairGrid(fifa_new)
p = p.map_upper(plt.scatter)
p = p.map_lower(sns.kdeplot, cmap="Blues_d")
p = p.map_diag(sns.kdeplot, lw=3, legend=False)
st.pyplot(p.figure)
plt.close(p.figure)

# JointGrid
st.subheader("Seaborn `JointGrid()`")
st.markdown("""
- `JointGrid` draws bivariate plots with marginal univariate plots.
""")
j = sns.JointGrid(x="Overall", y="Potential", data=fifa)
j = j.plot(sns.regplot, sns.histplot)
st.pyplot(j.figure)
plt.close(j.figure)

j = sns.JointGrid(x="Age", y="Potential", data=fifa)
j = j.plot(sns.regplot, sns.histplot)
st.pyplot(j.figure)
plt.close(j.figure)

st.markdown("Separate joint and marginal plots:")
j = sns.JointGrid(x="Overall", y="Potential", data=fifa)
j = j.plot_joint(plt.scatter, color=".5", edgecolor="w")
j = j.plot_marginals(sns.histplot, kde=False, color=".5")
st.pyplot(j.figure)
plt.close(j.figure)

st.markdown("No space between axes:")
j = sns.JointGrid(x="Overall", y="Potential", data=fifa, space=0)
j = j.plot_joint(sns.kdeplot, cmap="Blues_d")
j = j.plot_marginals(sns.kdeplot, fill=True)
st.pyplot(j.figure)
plt.close(j.figure)

st.markdown("Smaller plot with larger marginals:")
j = sns.JointGrid(x="Overall", y="Potential", data=fifa, height=5, ratio=2)
j = j.plot_joint(sns.kdeplot, cmap="Reds_d")
j = j.plot_marginals(sns.kdeplot, color="r", fill=True)
st.pyplot(j.figure)
plt.close(j.figure)

# Controlling plot size
st.subheader("Controlling the size and shape of the plot")
st.markdown("""
- `regplot()` size is controlled via Matplotlib figure.
""")
fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(x="Overall", y="Potential", data=fifa, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown("`lmplot()` size is controlled via FacetGrid:")
g = sns.lmplot(x="Overall", y="Potential", col="Preferred Foot", data=fifa, col_wrap=2, height=5, aspect=1)
st.pyplot(g.figure)
plt.close(g.figure)

# Heatmap
st.subheader("Correlation Heatmap")
st.markdown("""
- A heatmap visualizes the correlation matrix of numerical variables.
- Values range from -1 (negative correlation) to 1 (positive correlation), with 0 indicating no correlation.
""")

# Select numerical columns for correlation
numerical_cols = ['Age', 'Overall', 'Potential', 'Stamina', 'Strength']
corr_matrix = fifa[numerical_cols].corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix,  annot=True, cmap="coolwarm",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,ax=ax)
plt.title("Correlation Heatmap of Numerical Variables")
st.pyplot(fig)
plt.close(fig)

st.markdown("""
- **Observations**: 
  - Strong positive correlations appear in red.
  - Strong negative correlations appear in blue.
  - Check for relationships between variables like `Overall` and `Potential`.
""")

# Final image display
st.image(fifa_image, caption='FIFA 2019')
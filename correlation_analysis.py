#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

# 讀取資料
ftir_data_sheet = pd.read_excel('C:/Users/fishd/Desktop/Github/FTIR_dust/dataset/土壤DATA/訓練用/ALL_4000_750_881筆(未調).xlsx')
ftir_data_sheet=ftir_data_sheet.iloc[:, 1:]

# Plot the distribution of TOC(%)
plt.figure(figsize=(10, 6))
plt.hist(ftir_data_sheet['TOC(%)'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of TOC(%)')
plt.xlabel('TOC(%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#%%


# Plot the distribution of TOC(%)
plt.figure(figsize=(10, 6))
plt.hist(ftir_data_sheet['TOC(%)'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of TOC(%)')
plt.xlabel('TOC(%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# %%
# Compute the correlation between TOC(%) and each wavenumber's intensities
correlations = ftir_data_sheet.corr()['TOC(%)'][:-1]  # Exclude TOC(%) from the correlation with itself

# Plot the correlation coefficients
plt.figure(figsize=(15, 6))
plt.plot(correlations.index.astype(float), correlations.values, color='blue')
plt.title('Correlation between TOC(%) and FTIR Intensities at Different Wavenumbers')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Correlation Coefficient')
plt.grid(True)
plt.show()

# Display the highest and lowest correlation coefficients
correlations_sorted = correlations.sort_values(ascending=False)
highest_correlations = correlations_sorted.head(5)
lowest_correlations = correlations_sorted.tail(5)

highest_correlations, lowest_correlations

# %%
# Select samples with high and low TOC(%) values
high_toc_samples = ftir_data_sheet.nlargest(5, 'TOC(%)')
low_toc_samples = ftir_data_sheet.nsmallest(5, 'TOC(%)')
all_toc_samples = ftir_data_sheet.sort_values('TOC(%)', ascending=False)
# Define wavenumbers
wavenumbers = ftir_data_sheet.columns[:-1].astype(float)

# Plot FTIR spectra for high TOC(%) samples
plt.figure(figsize=(15, 6))
for idx, row in high_toc_samples.iterrows():
    plt.plot(wavenumbers, row[:-1], label=f'High TOC(%) - Sample {idx+1}')

plt.title('FTIR Spectra for High TOC(%) Samples')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plot FTIR spectra for low TOC(%) samples
plt.figure(figsize=(15, 6))
for idx, row in low_toc_samples.iterrows():
    plt.plot(wavenumbers, row[:-1], label=f'Low TOC(%) - Sample {idx+1}')

plt.title('FTIR Spectra for Low TOC(%) Samples')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (%)')
plt.legend()
plt.grid(True)
plt.show()



# %%
# Define significant wavenumber ranges for organic compounds
significant_regions_a = {
    '1a': (650, 820),
    '2a': (850, 1220),
    '3a': (1250, 1750),
    '4a': (2800, 3000),
}
significant_regions_b = {

    '1b': (740, 1146),
}

# Plot FTIR spectra for high TOC(%) samples with highlighted regions
plt.figure(figsize=(15, 6))
for idx, row in high_toc_samples.iterrows():
    plt.plot(wavenumbers, row[:-1], label=f'High TOC(%) - Sample {idx+1}')

# Highlight regions outside significant ranges
for start, end in significant_regions_a.values():
    plt.axvspan(start, end, color='white', alpha=0)  # Set to transparent for significant regions

# Highlight the areas outside the significant regions with yellow
plt.axvspan(820, 850, color='yellow', alpha=0.3)  # Between 1a and 2a
plt.axvspan(1220, 1250, color='yellow', alpha=0.3)  # Between 2a and 3a
plt.axvspan(1750, 2800, color='yellow', alpha=0.3)  # Between 3a and 4a
plt.axvspan(3000, max(wavenumbers), color='yellow', alpha=0.3)  # Right of the last range

plt.title('FTIR Spectra for High TOC(%) Samples with Highlighted Non-Organic Compound Regions')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Plot FTIR spectra for low TOC(%) samples with highlighted regions
plt.figure(figsize=(15, 6))
for idx, row in low_toc_samples.iterrows():
    plt.plot(wavenumbers, row[:-1], label=f'Low TOC(%) - Sample {idx+1}')

# Highlight regions outside significant ranges
for start, end in significant_regions_a.values():
    plt.axvspan(start, end, color='white', alpha=0)  # Set to transparent for significant regions

# Highlight the areas outside the significant regions with yellow

plt.axvspan(820, 850, color='yellow', alpha=0.3)  # Between 1a and 2a
plt.axvspan(1220, 1250, color='yellow', alpha=0.3)  # Between 2a and 3a
plt.axvspan(1750, 2800, color='yellow', alpha=0.3)  # Between 3a and 4a
plt.axvspan(3000, max(wavenumbers), color='yellow', alpha=0.3)  # Right of the last range

plt.title('FTIR Spectra for Low TOC(%) Samples with Highlighted Non-Organic Compound Regions')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Plot FTIR spectra for all TOC(%) samples with highlighted regions
plt.figure(figsize=(15, 6))
for idx, row in all_toc_samples.iterrows():
    plt.plot(wavenumbers, row[:-1], label=f'All TOC(%) - Sample {idx+1}')

# Highlight regions outside significant ranges
for start, end in significant_regions_a.values():
    plt.axvspan(start, end, color='white', alpha=0)  # Set to transparent for significant regions

# Highlight the areas outside the significant regions with yellow

plt.axvspan(820, 850, color='yellow', alpha=0.3)  # Between 1a and 2a
plt.axvspan(1220, 1250, color='yellow', alpha=0.3)  # Between 2a and 3a
plt.axvspan(1750, 2800, color='yellow', alpha=0.3)  # Between 3a and 4a
plt.axvspan(3000, max(wavenumbers), color='yellow', alpha=0.3)  # Right of the last range

plt.title('FTIR Spectra for All TOC(%) Samples with Highlighted Non-Organic Compound Regions')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# %%

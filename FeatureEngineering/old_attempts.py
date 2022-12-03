"""
# Attempt 1 - Abandoned.
# Following code makes score much worse and volatile (many high negatives).

# Normalize to range 0-1
df['CITY'] = (df['CITY'] - df['CITY'].min()) / (df['CITY'].max() - df['CITY'].min())
df['Zip Code'] = (df['Zip Code'] - df['Zip Code'].min()) / (df['Zip Code'].max() - df['Zip Code'].min())

# Normalize with z-score
df['total_population'] = abs((df['total_population'] - df['total_population'].mean())/(df['total_population'].std()))
df['white_Population'] = abs((df['white_Population'] - df['white_Population'].mean())/(df['white_Population'].std()))
df['black_Population'] = abs((df['black_Population'] - df['black_Population'].mean())/(df['black_Population'].std()))
df['asian_Population'] = abs((df['asian_Population'] - df['asian_Population'].mean())/(df['asian_Population'].std()))
df['hispanic_Population'] = abs((df['hispanic_Population'] - df['hispanic_Population'].mean())/(df['hispanic_Population'].std()))

df['ALAND10'] = abs((df['ALAND10'] - df['ALAND10'].mean())/(df['ALAND10'].std()))
df['AWATER10'] = abs((df['AWATER10'] - df['AWATER10'].mean())/(df['AWATER10'].std()))
df['AREA'] = abs((df['AREA'] - df['AREA'].mean())/(df['AREA'].std()))
df['Per Capita Income'] = abs((df['Per Capita Income'] - df['Per Capita Income'].mean())/(df['Per Capita Income'].std()))

#df['median_rent'] = abs((df['median_rent'] - df['median_rent'].mean())/(df['median_rent'].std()))
print(df.sample(10))
"""

# Makes score consistently, substantially lower.
#df['ALAND10'] = (df['ALAND10'] - df['ALAND10'].min()) / (df['ALAND10'].max() - df['ALAND10'].min()) * df['median_rent'].median()
#df['AWATER10'] = (df['AWATER10'] - df['AWATER10'].min()) / (df['AWATER10'].max() - df['AWATER10'].min()) * df['median_rent'].median()

# Makes score much more volatile (super high negatives, sometimes positives)
#df['ALAND10'] = abs((df['ALAND10'] - df['ALAND10'].mean()) / (df['ALAND10'].std()))
#df['AWATER10'] = abs((df['AWATER10'] - df['AWATER10'].mean()) / (df['AWATER10'].std()))

"""
# Normalize population with z-score
# Makes score substantially worse.
df['total_population'] = abs((df['total_population'] - df['total_population'].mean()) / (df['total_population'].std()))
df['white_Population'] = abs((df['white_Population'] - df['white_Population'].mean()) / (df['white_Population'].std()))
df['black_Population'] = abs((df['black_Population'] - df['black_Population'].mean()) / (df['black_Population'].std()))
df['asian_Population'] = abs((df['asian_Population'] - df['asian_Population'].mean()) / (df['asian_Population'].std()))
df['hispanic_Population'] = abs((df['hispanic_Population'] - df['hispanic_Population'].mean()) / (df['hispanic_Population'].std()))
"""

"""
# Normalize with range 0 - median of target.
# Seems to have no effect on score.
df['total_population'] = (df['total_population'] - df['total_population'].min()) / (df['total_population'].max() - df['total_population'].min()) * df['median_rent'].median()
df['white_Population'] = (df['white_Population'] - df['white_Population'].min()) / (df['white_Population'].max() - df['white_Population'].min()) * df['median_rent'].median()
df['black_Population'] = (df['black_Population'] - df['black_Population'].min()) / (df['black_Population'].max() - df['black_Population'].min()) * df['median_rent'].median()
df['asian_Population'] = (df['asian_Population'] - df['asian_Population'].min()) / (df['asian_Population'].max() - df['asian_Population'].min()) * df['median_rent'].median()
df['hispanic_Population'] = (df['hispanic_Population'] - df['hispanic_Population'].min()) / (df['hispanic_Population'].max() - df['hispanic_Population'].min()) * df['median_rent'].median()
df['total_population'] = abs((df['total_population'] - df['total_population'].mean()) / (df['total_population'].std()))
"""
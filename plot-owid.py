import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


owid = pd.read_excel('./data/owid-covid-data.xlsx', index_col=0,usecols="A,C:J,AI:AN")
owid.set_index('date',inplace=True)

owidWord = owid.loc[owid['location'] == 'World']
owidWord.plot('date',['new_cases_smoothed','new_deaths_smoothed','new_vaccinations'],
               kind='line',figsize=(6,3),title='OWID world covid data',logy=True,rot=45)
plt.tight_layout()
plt.savefig('./img/owid-world.pdf')

# owidCountry = owid.loc[owid['iso_code'].isin(['IRN', 'USA', 'GBR', 'FRA', 'CHN'])]
owidCountry = owid.loc[owid['location'].isin(['Iran', 'United States', 'United Kingdom', 'France', 'China','World'])]
owidCountry.groupby('location')['new_cases_smoothed'].plot(
    kind='line',figsize=(5,4),title='OWID world covid data',logy=True,rot=45,legend=True)
plt.legend(loc="lower center", bbox_to_anchor=(0.5,1.1), ncol=3)
plt.tight_layout()
plt.savefig('./img/owid-new-case-country-6.pdf')
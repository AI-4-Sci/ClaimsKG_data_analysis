import os
import pandas as pd
from tqdm import tqdm
import ast
print(os.getcwd())
tqdm.pandas()
#%%
claimskg_df = pd.concat([pd.read_csv(fn, sep='\t') for fn in os.listdir() if "output_got" in fn])
del claimskg_df['Unnamed: 0']
claimskg_df.info()

claimskg_df['related_links'] = claimskg_df['extra_referred_links'].apply(lambda x: x.split(":-:"))
claimskg_df['has_links'] = claimskg_df['related_links'].apply(lambda x: len(x) > 0)


#%%
class DomainAnnotator:
    def __init__(self, domains, column):
        self.domains = domains
        self.column = column

    def annotate(self, domains):
        matches = []

        for domain in domains:
            for sci_domain in self.domains:
                if sci_domain in domain:
                    if domain == sci_domain:
                        matches.append(domain)
                        # check if subdomain is below sci_subdomain
                    elif domain.endswith('.' + sci_domain):
                        matches.append(domain)

        return matches

    def process(self, fcs):
        print(len(fcs))
        fcs['sci_domains'] = fcs[self.column].progress_apply(self.annotate)
        fcs["has_sci_url"] = fcs['sci_domains'].apply(lambda x: True if len(x) > 0 else False)
        return fcs

class SciRepoDomainAnnotator(DomainAnnotator):
    def __init__(self):
        super().__init__(domains=pd.read_csv('subdomains_2.csv')['domain'].tolist(), column='related_links')

#%%
import multiprocessing
import pandas as pd
import numpy as np

def mp_func(data, func, workers=10, splitted=False, concat=False):

    """if not splitted:
        if type(df) == list:
            if type(df[0]) == pd.DataFrame:
                print('data is already splitted into multiple dataframes')
                splitted_data = df
                if len(splitted_data) != workers:
                    workers = len(splitted_data)
                    print('num workers does not fit to number of splits. Changed num workers')


        elif type(df) == pd.DataFrame:
            splitted_data = np.array_split(df, workers)

        elif type(df) == pd.Series:
            splitted_data = np.array_split(df, workers)

        else:
            print('unknown data type. Return data as is')
            return df"""

    if not splitted:
        data = np.array_split(data, workers)
    try:
        pool = multiprocessing.Pool(processes=workers)
        results = pool.map(func, [split for split in data])

        if concat:
            results = pd.concat(results, ignore_index=False)

    except KeyboardInterrupt:
        pool.terminate()

    pool.terminate()

    return results

#%%
scirepo_da = SciRepoDomainAnnotator()
claimskg_df_annotated = mp_func(claimskg_df, scirepo_da.process, 80, False, True)
claimskg_df_annotated.to_csv('claimskg_df_annotated.tsv', sep='\t', index=False)

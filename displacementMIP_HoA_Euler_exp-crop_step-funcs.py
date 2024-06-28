"""
Created in April 2024
Updated in June 2024

description: Load hazard, pre-processed exposure matching 2 sets of impact functions,
            the 2 sets of impact functions and calculate impacts for the Horn of
            Africa countries (Ethiopia, Somalia, Sudan). 
            Incl. simple approach using a 10cm flood threshold step function

@author: simonameiler
"""
import os
import sys
import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
from pathlib import Path
import copy as cp

from climada.hazard import Hazard
from climada.entity.exposures import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc
from climada.util import coordinates as u_coords


sys.path.append('/cluster/project/climate/meilers/scripts/displacement/global-displacement-risk')
import exposure, vulnerability

def main(country):
    
    country = str(country)
    
    disp_code_base = Path('/cluster/project/climate/meilers/scripts/displacement/global-displacement-risk/')
    
    # load hazard
    HAZ_TYPE = 'FL'
    HAZ_FOLDER = Path(f'/cluster/work/climate/evelynm/IDMC_UNU/hazard/flood_HM_CIMA/{country}/HISTORICAL/')
    
    haz_files = np.sort([str(file) for file in HAZ_FOLDER.glob('*.tif')]).tolist()
    rp = np.array([int(Path(file).stem[-4:]) for file in haz_files])
    
    haz = Hazard.from_raster(
        haz_type=HAZ_TYPE, files_intensity=haz_files, src_crs='WGS84',
        attrs={'unit': 'm', 'event_id': np.arange(len(haz_files)), 'frequency':1/rp}
    )
    
    # convert intensity given in cm to m
    haz.intensity = haz.intensity/100

    
    # load exposures
    # ### BEM subcomponents - prep for use with IVM impact functions

    # Load the full dataframe, without further re-aggregation / processing other than adding centroids
    gdf_bem_subcomps = exposure.gdf_from_bem_subcomps(country, opt='full')

    # filter and apply impf id
    gdf_bem_subcomps = gdf_bem_subcomps[gdf_bem_subcomps.valhum>0.001] # filter out rows with basically no population
    gdf_bem_subcomps['impf_FL'] = gdf_bem_subcomps.apply(lambda row: vulnerability.DICT_PAGER_FLIMPF_IVM[row.se_seismo], axis=1)

    # replace impf 3 --> 5 for 2-storeys and more
    gdf_bem_subcomps.loc[((gdf_bem_subcomps.bd_3_floor+gdf_bem_subcomps.bd_2_floor)>0.5)
                         &(gdf_bem_subcomps.impf_FL==3), "impf_FL"] = 5

    # replace impf 4 --> 6 for 2-storeys and more
    gdf_bem_subcomps.loc[((gdf_bem_subcomps.bd_3_floor+gdf_bem_subcomps.bd_2_floor)>0.5)
                         &(gdf_bem_subcomps.impf_FL==4), "impf_FL"] = 6

    # remove for now unnecessary cols and prepare gdf for CLIMADA Exposure
    gdf_bem_subcomps.rename({'valhum' : 'value'}, axis=1)
    for col in ['iso3', 'sector', 'valfis', 'se_seismo']:
        gdf_bem_subcomps.pop(col)
    gdf_bem_subcomps

    # Make CLIMADA Exposure with mutliply defined centroids
    exp_bem = Exposures(gdf_bem_subcomps)
    exp_bem.gdf.rename({'valhum': 'value'}, axis=1, inplace=True)
    exp_bem.value_unit = 'Pop. count'
    exp_bem.gdf['longitude'] = exp_bem.gdf.geometry.x
    exp_bem.gdf['latitude'] = exp_bem.gdf.geometry.y
    exp_bem.gdf = exp_bem.gdf[~np.isnan(
        exp_bem.gdf.latitude)]  # drop nan centroids

    cntry_iso = u_coords.country_to_iso(country)
    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])
    polygon_gdf = gpd.GeoDataFrame([1], geometry=[geom_cntry], crs=gdf_bem_subcomps.crs)
    overlay_bem = gpd.sjoin(gdf_bem_subcomps, polygon_gdf, how="inner", op="intersects")

    exp_ivm = Exposures(overlay_bem)

    # ### BEM subcomponents - prep for use with IVM impact functions

    # Load the full dataframe, without further re-aggregation / processing other than adding centroids
    gdf_bem_subcomps = exposure.gdf_from_bem_subcomps(country, opt='full')

    # filter and apply impf id
    gdf_bem_subcomps = gdf_bem_subcomps[gdf_bem_subcomps.valhum>0.001] # filter out rows with basically no population
    gdf_bem_subcomps['impf_FL'] = gdf_bem_subcomps.apply(lambda row: vulnerability.DICT_PAGER_FLIMPF_CIMA[row.se_seismo], axis=1)

    # remove for now unnecessary cols and prepare gdf for CLIMADA Exposure
    gdf_bem_subcomps.rename({'valhum' : 'value'}, axis=1)
    for col in ['iso3', 'sector', 'valfis', 'se_seismo', 'bd_1_floor','bd_2_floor','bd_3_floor']:
        gdf_bem_subcomps.pop(col)
        
    # Make CLIMADA Exposure with mutliply defined centroids

    exp_bem = Exposures(gdf_bem_subcomps)
    exp_bem.gdf.rename({'valhum': 'value'}, axis=1, inplace=True)
    exp_bem.value_unit = 'Pop. count'
    exp_bem.gdf['longitude'] = exp_bem.gdf.geometry.x
    exp_bem.gdf['latitude'] = exp_bem.gdf.geometry.y
    exp_bem.gdf = exp_bem.gdf[~np.isnan(
        exp_bem.gdf.latitude)]  # drop nan centroids

    cntry_iso = u_coords.country_to_iso(country)
    geom_cntry = shapely.ops.unary_union(
        [geom for geom in
         u_coords.get_country_geometries([cntry_iso]).geometry])
    polygon_gdf = gpd.GeoDataFrame([1], geometry=[geom_cntry], crs=gdf_bem_subcomps.crs)
    overlay_bem = gpd.sjoin(gdf_bem_subcomps, polygon_gdf, how="inner", op="intersects")

    exp_cima = Exposures(overlay_bem)
    
    
    #Impact functions
    # IVM approach
    
    impf_set_ivm = vulnerability.IMPF_SET_FL_IVM
    impf_set_cima = vulnerability.IMPF_SET_FL_CIMA
    
    rps = rp.tolist()
        
    def calculate_impacts_and_return_df(exp, impf_set, haz, threshold):
        impf_set_step = ImpactFuncSet()
        for imp_id in impf_set.get_ids(haz_type='FL'):
            impf_set.get_func(fun_id=imp_id)
            y = impf_set.get_func(fun_id=imp_id)[0].intensity
            x = impf_set.get_func(fun_id=imp_id)[0].mdd
            flood_thres = np.interp(threshold, x, y)
            impf_set_step.append(
                ImpactFunc.from_step_impf(
                    intensity=(0, flood_thres, flood_thres * 10),
                    haz_type='FL',
                    impf_id=imp_id,
                    intensity_unit='m'
                )
            )
        impcalc = ImpactCalc(exp, impf_set_step, haz)
        impact = impcalc.impact()
        aai_agg = impact.aai_agg  # Annual average displacement
        pm_data = impact.at_event.tolist()
        # Total population calculation
        total_population = exp.gdf['value'].sum()  # Assuming 'value' is the column with population data
    
        # Create DataFrame with AAI, PMD data, and total population
        data = {
            'AAI': [aai_agg] * len(rps),
            'PMD': pm_data,
            'Threshold': [threshold] * len(rps),
            'Return Period': rps,
            'Total Population': [total_population] * len(rps)  # Adding total population
        }
        return pd.DataFrame(data)

    all_dfs = []
    thresholds = [0.3, 0.55, 0.7]
    for threshold in thresholds:
        df = calculate_impacts_and_return_df(exp_cima, impf_set_cima, haz, threshold)
        all_dfs.append(df)
        df_ivm = calculate_impacts_and_return_df(exp_ivm, impf_set_ivm, haz, threshold)
        all_dfs.append(df_ivm)
    
    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined DataFrame to CSV
    file_path = disp_code_base.joinpath(f'{country}_exp-crop_displacement.csv')
    print(f"Saving file to: {file_path}")
    combined_df.to_csv(file_path, index=False)

    ##############################################################################
    ## stepfunction at 10cm - it doesn't matter which exposure we take
    exp_base = cp.deepcopy(exp_cima)
    exp_base.gdf['impf_FL'] = 1
    
    # get stepfunction
    thresh = 0.1 # set threshold flood depth for displacement
    impf_step = ImpactFunc.from_step_impf(intensity=(0, thresh, 20), haz_type='FL', impf_id=1)
    imp_fun_set = ImpactFuncSet([impf_step])
    
    # calculate impact
    impcalc = ImpactCalc(exp_base, imp_fun_set, haz)
    impact_simple = impcalc.impact()
    aai_agg = impact_simple.aai_agg
    pm_data = impact_simple.at_event.tolist()
    data = {
        'AAI': [aai_agg] * len(rps),
        'PMD': pm_data,
        'Return Period': rps}
    
    df_simple = pd.DataFrame(data)
    
    # Save the combined DataFrame to CSV
    file_path = disp_code_base.joinpath(f'{country}_exp-crop_displacement_10cm.csv')
    print(f"Saving file to: {file_path}")
    df_simple.to_csv(file_path, index=False)

if __name__ == "__main__":
    main(*sys.argv[1:])
import openml
import pandas as pd

def get_filtered_cc18_ids():
    # OpenML-CC18 suite ID is 99
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    
    # We should iterate through the tasks in the suite and get their details
    # list_tasks with task_ids might be failing because of API change or limitation?
    # Error said: Found illegal filter(s): task_ids
    # But list_tasks doc says it supports task_ids list. Maybe the list is too long for URL?
    
    # Let's try to get all tasks with tag 'OpenML-CC18' instead?
    # Or just fetch tasks one by one (slow but safe) or in batches?
    
    # Actually, list_tasks supports tag.
    tasks = openml.tasks.list_tasks(
        tag='OpenML-CC18',
        output_format="dataframe"
    )
    
    # Filter for classification
    # CC18 is mostly classification, but let's be sure
    # In list_tasks, task_type_id 1 is Supervised Classification
    
    # We need dataset metadata to filter by size and features
    # This might be slow if we fetch one by one.
    # Luckily list_tasks returns some metadata.
    
    filtered_ids = []
    
    print(f"Total tasks in CC18: {len(tasks)}")
    
    for _, task in tasks.iterrows():
        # LoCalPFN filters:
        # num_instances: 2000 - 200,000
        # num_features: <= 100
        # num_classes: <= 10
        
        # task object has 'NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses'
        # Note: column names might differ in dataframe format
        
        n_instances = task['NumberOfInstances']
        n_features = task['NumberOfFeatures']
        n_classes = task['NumberOfClasses']
        
        if n_instances < 2000 or n_instances > 200000:
            continue
        if n_features > 100:
            continue
        if n_classes > 10:
            continue
            
        # Also exclude the ones explicitly mentioned in LoCalPFN code if they are in CC18
        # "openml__cjs__14967", "openml__higgs__146606", "openml__jm1__3904", "openml__sick__3021"
        # We only have dataset IDs here.
        d_id = task['did']
        
        filtered_ids.append(int(d_id))
        
    print(f"Filtered datasets matching LoCalPFN criteria: {len(filtered_ids)}")
    print(f"IDs: {filtered_ids}")
    return filtered_ids

if __name__ == "__main__":
    ids = get_filtered_cc18_ids()

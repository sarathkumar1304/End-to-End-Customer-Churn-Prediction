import pandas as pd
from zenml import step

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    data = {
        'Age':[30,58], 
        'Gender':[0,1],
        'Tenure':[39,38], 
        'Usage Frequency':[14,21],
        'Support Calls':[5,7], 
        'Payment Delay':[18,7], 
        'Subscription Type':[3,1],
       'Contract Length':[2,1], 
       'Total Spend':[10,12],
       'Last Interaction':[2,1]
    }
    df = pd.DataFrame(data)
    json_data = df.to_json(orient="split")
    return json_data

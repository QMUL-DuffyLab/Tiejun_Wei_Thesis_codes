# GCN_Net
FIRST THING FIRST: unpack the zip file in ./data
the data is packed because there are too many MD data points.
note that the Github is not allowed to store large files. If you need the testing data please contact us: c.duffy@qmul.ac.uk

A new repo with cleaned up code from LHCII_spatial project

Use networkx_gen etc to generate pytorch objects like data point from MD snapshots(output mol2 files)

Use GCN_regression.py to train the model using model defined in the script.

A visualization try is listed here but some bug may still persist: embedding_visualization.ipynb

Also you can use GNN_explainer.ipynb to interpret the explainability of the GCN.
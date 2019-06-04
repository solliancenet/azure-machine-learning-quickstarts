# Pick your Python Interpreter: 'azure_automl':conda
# 
# Run each cell to complete the setup

#%%
get_ipython().run_cell_magic('sh', '', 'pip install --upgrade azureml-sdk[notebooks,explain,automl,contrib]')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install --upgrade Keras')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install --upgrade tensorflow')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install onnxmltools')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install tf2onnx')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install keras2onnx')


#%%
get_ipython().run_cell_magic('sh', '', 'pip install onnxruntime')


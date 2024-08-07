# Databricks notebook source
# MAGIC %md
# MAGIC # Minimal Example of Serving R Models on Databricks
# MAGIC
# MAGIC Model serving doesn't directly support R and the main entry point is python. These two challenges can be overcome by:
# MAGIC 1. Installing R within the serving container
# MAGIC 2. Using a PyFunc with `rpy2` to load and use the R model
# MAGIC
# MAGIC The purpose is to demonstrate that it's possible with a simple model and the minimum required changes.
# MAGIC This is not intended to show a generic way to serve R models, that is still being explored
# MAGIC
# MAGIC _Tested with runtime 14.3 LTS_

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Requirements
# MAGIC
# MAGIC - Python: `rpy2`
# MAGIC - R: `carrier`

# COMMAND ----------

# MAGIC %python
# MAGIC %pip install -q rpy2 databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# this is just a trick to dramatically improve install time of R packages
options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])))
release <- system("lsb_release -c --short", intern = T)
options(repos = c(POSIT = paste0("https://packagemanager.posit.co/cran/__linux__/", release, "/latest")))

# COMMAND ----------

install.packages(c("mlflow", "carrier"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Simple R Model
# MAGIC
# MAGIC In this case the model will just a simple linear model (`lm()`). In practice this can be as complex as required.

# COMMAND ----------

# example data
x <- 1:10
y <- 2 * x + 1

# create linear model
lm_model <- lm(y ~ x)

# COMMAND ----------

# example of invoking our simple model
predict(lm_model, data.frame(x = 5:25))

# COMMAND ----------

# use `carrier` package to serialise the model into a self-contained function
#The crate function in the carrier package is a tool that helps wrap and construct the R model, making it a crated function
#This is then passed in to be logged as an mlflow model

model <- carrier::crate(function(x) {
  yhat <- stats::predict(
    object = !!lm_model,
    new = data.frame(x = x)
  )
  unname(yhat)
})

# COMMAND ----------

# MAGIC %md
# MAGIC Create an experiment run and log the R model

# COMMAND ----------

library(mlflow)

run <- mlflow_start_run()
mlflow_log_model(model = model, artifact_path = "crate")
mlflow_end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC We're only interested in the `crate.bin` artifact that's created, this is what we'll extract and add to our pyfunc as an artifact.
# MAGIC
# MAGIC For now we need to get the absolute path for the artifact and pass this to `mlflow.pyfunc.log_model` later.

# COMMAND ----------

# in this case, the path we need to copy for later is...
cat(file.path(run$artifact_uri, "crate", "crate.bin"))

# COMMAND ----------

# MAGIC %python
# MAGIC # set the location now
# MAGIC _CRATE_LOCATION = "dbfs:/databricks/mlflow-tracking/1857178648934777/ad87f24718ae4b89a149e0857fa29f1c/artifacts/crate/crate.bin"

# COMMAND ----------

# MAGIC %md
# MAGIC We know what we expect the input and output schema to be and therefore we can craft a PyFunc that will work for this model.
# MAGIC
# MAGIC ## Building a PyFunc
# MAGIC
# MAGIC `carrier` writes out its contents via `base::saveRDS()`, therefore when using `rpy2` we'll directly use `base::readRDS()`.
# MAGIC
# MAGIC We know that our inputs are strictly numeric for this model and will ensure all inputs are `FloatVector`.
# MAGIC

# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC from rpy2 import robjects
# MAGIC from rpy2.robjects import r, FloatVector
# MAGIC from rpy2.robjects.packages import importr
# MAGIC
# MAGIC class CarrierServing(mlflow.pyfunc.PythonModel):
# MAGIC   
# MAGIC   def __init__(self):
# MAGIC     pass
# MAGIC     
# MAGIC   def load_context(self, context):
# MAGIC     # load the crate saved by carrier package
# MAGIC     base = importr('base')
# MAGIC     self.crate = base.unserialize(base.readRDS(context.artifacts["crate"]))
# MAGIC
# MAGIC   def predict(self, context, model_input, params=None):
# MAGIC     yhat = self.crate(FloatVector(model_input['x']))
# MAGIC     return list(yhat)

# COMMAND ----------

# MAGIC %md
# MAGIC Explicitly define the model signature - this is a requirement to serve models.

# COMMAND ----------

# MAGIC %python
# MAGIC from mlflow.models import ModelSignature
# MAGIC from mlflow.types.schema import Schema, ColSpec
# MAGIC
# MAGIC # construct the signature object (manual method)
# MAGIC input_schema = Schema([ColSpec("double", "x")])
# MAGIC output_schema = Schema([ColSpec("double")])
# MAGIC
# MAGIC signature = ModelSignature(inputs=input_schema, outputs=output_schema)
# MAGIC
# MAGIC # manually craft input examples
# MAGIC input_example = {"x": [0, 1, 2, 10]}

# COMMAND ----------

# MAGIC %md
# MAGIC In order to ensure that R is added to the container we need to override the `conda_env` that is specified.
# MAGIC
# MAGIC In this particular example the versions of languages and libraries are all hard-coded. Special attention is required to ensure this matches requirements. 
# MAGIC
# MAGIC The special part here is adding R and other entries to `"dependencies"`.
# MAGIC
# MAGIC Ideally this would by dynamic, but this is again trying to show what's possible, not perfect.

# COMMAND ----------

# MAGIC %python
# MAGIC # hand craft conda.yaml dict, can automate but easier for now
# MAGIC adjusted_conda_env = {
# MAGIC     "name": "mlflow-env",
# MAGIC     "channels": ["conda-forge"],
# MAGIC     "dependencies": [
# MAGIC         "libstdcxx-ng=12",
# MAGIC         "r-base=4.4.1",
# MAGIC         "python=3.10.12",
# MAGIC         "pip<=24.0",
# MAGIC         {
# MAGIC             "pip": [
# MAGIC                 "pandas",
# MAGIC                 "rpy2",
# MAGIC                 "cloudpickle==2.0.0",
# MAGIC                 "mlflow==2.9.2"
# MAGIC             ],
# MAGIC         },
# MAGIC     ],
# MAGIC }

# COMMAND ----------

# MAGIC %python
# MAGIC with mlflow.start_run():
# MAGIC   model_info = mlflow.pyfunc.log_model(
# MAGIC       artifact_path="model",
# MAGIC       python_model=CarrierServing(),
# MAGIC       input_example=input_example,
# MAGIC       conda_env=adjusted_conda_env,
# MAGIC       signature=signature,
# MAGIC       artifacts = {'crate': _CRATE_LOCATION}
# MAGIC   )

# COMMAND ----------

# MAGIC %python
# MAGIC # Load the model from the tracking server and perform inference
# MAGIC model = mlflow.pyfunc.load_model(model_info.model_uri)
# MAGIC model.predict({"x": [float(x) for x in range(10)]})

# COMMAND ----------

# MAGIC %python
# MAGIC import pandas as pd
# MAGIC # can predict with pandas df
# MAGIC ex_df = pd.DataFrame(data={"x": [float(x) for x in range(10)]})
# MAGIC model.predict(data = ex_df)

# COMMAND ----------

# MAGIC %md ## Register Model to Unity Catalog

# COMMAND ----------

# MAGIC %python
# MAGIC _MODEL_NAME = 'main.chandhana.r_linear_model'

# COMMAND ----------

# MAGIC %python
# MAGIC import mlflow
# MAGIC catalog = "main"
# MAGIC schema = "chandhana"
# MAGIC model_name = "r_linear_model"
# MAGIC # registry the model into UC schema
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC mlflow.register_model(
# MAGIC     model_uri="runs:/1530b20cd42e46188f9bdf846fda3a43/model",
# MAGIC     name=f"{catalog}.{schema}.{model_name}"
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can then navigate via UI to the model and serve as usual

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""

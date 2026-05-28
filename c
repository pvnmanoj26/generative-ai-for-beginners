[1mdiff --git a/06-text-generation-apps/python/Dockerfile b/06-text-generation-apps/python/Dockerfile[m
[1mindex 13ec67ac..fe29a5d2 100644[m
[1m--- a/06-text-generation-apps/python/Dockerfile[m
[1m+++ b/06-text-generation-apps/python/Dockerfile[m
[36m@@ -10,8 +10,8 @@[m [mRUN apt-get update && apt-get install -y \[m
 COPY requirements_cloud.txt .[m
 RUN pip install --no-cache-dir -r requirements_cloud.txt[m
 [m
[31m-COPY clinical_app_cloud_v01.py .[m
[32m+[m[32mCOPY clinical_api.py .[m
 [m
 EXPOSE 8080[m
 [m
[31m-CMD ["python", "clinical_app_cloud_v01.py"][m
[32m+[m[32mCMD ["uvicorn", "clinical_api:app", "--host", "0.0.0.0", "--port", "8080"][m

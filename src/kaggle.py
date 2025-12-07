import kagglehub

# Download Pneumonia dataset
print("Downloading Pneumonia Dataset...")
path_pneumonia = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to Pneumonia dataset files:", path_pneumonia)

# Download COVID-19 dataset
print("\nDownloading COVID-19 Dataset...")
path_covid = kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")
print("Path to COVID-19 dataset files:", path_covid)

# Download Additional Chest X-Ray Dataset (Covid, Pneumonia, Normal)
print("\nDownloading Additional Dataset (Prashant268)...")
path_additional = kagglehub.dataset_download("prashant268/chest-xray-covid19-pneumonia")
print("Path to Additional dataset files:", path_additional)

# Download NIH Chest X-ray Dataset (Sample) - Adult Data
print("\nDownloading NIH Chest X-ray Dataset (Sample)...")
path_nih = kagglehub.dataset_download("nih-chest-xrays/sample")
print("Path to NIH dataset files:", path_nih)

# Download COVID-19 Chest X-Ray Dataset (Bachrr)
print("\nDownloading COVID-19 Chest X-Ray Dataset (Bachrr)...")
path_covid_bachrr = kagglehub.dataset_download("bachrr/covid-chest-xray")
print("Path to Bachrr COVID-19 dataset files:", path_covid_bachrr)
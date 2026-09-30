# Data sources (Step 0.7)

Checked on 2026-10-01 from the institute server. "Verified" means the ID, URL or table was queried and returned the expected content.

| Dataset | ID / URL | Checked result | Status |
| --- | --- | --- | --- |
| GSED | `ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")` | Catalogue page live; availability 2017-01-01 to 2025 | Verified (catalogue); band check after Earth Engine registration |
| VIIRS night lights | `ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V22")`, band `average_masked` | Catalogue page live; availability 2012-04-01 to 2025, so one version covers 2018–2022 | Verified (catalogue); band check after registration |
| ESA WorldCover | `ESA/WorldCover/v200` (2021) | Catalogue page live | Verified (catalogue) |
| Municipal GDP | SIDRA table 5938, variables 37 (GDP), 498 (GVA total), 513 (agro), 517 (industry), 6575 (services excl. public), 525 (public admin), 543 (taxes); values in R$ 1,000 | Metadata returned; years 2002–2023; municipal level N6 available | Verified |
| Population | SIDRA table 4709 (Censo 2022), variable 93 "População residente" | Metadata returned | Verified |
| Municipalities and regions | IBGE Localidades API `/api/v1/localidades/estados/31/municipios` | 853 municipalities; 70 immediate and 13 intermediate regions | Verified |
| Municipal boundaries | `https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/MG/MG_Municipios_2022.zip` | HTTP 200 | Verified |
| OpenStreetMap | `https://download.geofabrik.de/south-america/brazil/sudeste-230101.osm.pbf` (snapshot 2023-01-01) | HTTP 200 | Verified |
| Caption imagery | Esri World Imagery (licence check) / Sentinel-2 fallback | Not yet checked | Step 4.1 |

Finding (2026-10-01): IBGE publishes only total GDP and GDP per capita for 2022-2023 at municipal level; value added by sector (agro, industry, services, public) stops at 2021 in both SIDRA 5938 and the FTP database (base_de_dados_2010_2023). Decision: the main year stays 2022 (GDP target, Census population). Sector shares come from 2021 (config study.sector_year) and are used only for the extreme-activity flags and the error analysis.

Pending: the Earth Engine band-level checks need the registered Cloud project (Step 0.2).

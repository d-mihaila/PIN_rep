{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9a7a937c-cbad-4417-be74-f9a4670291af",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import json\n",
    "from collections import defaultdict\n",
    "from datetime import datetime\n",
    "from acled import *\n",
    "from idmc_fetch import *\n",
    "from geolocation import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1dd5404c-1336-4676-b8d2-89419b3cc6c0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "We are looking specifically from 2022-06-22 to 2022-07-13.\n",
      "Found 3 unique locations:\n",
      "\n",
      "Toponym                        Latitude        Longitude      \n",
      "------------------------------------------------------------\n",
      "Pakistan                       30.0            70.0           \n",
      "Sindh                          26.13333        68.76667       \n",
      "Quetta                         30.18414        67.00141       \n",
      "[{'toponym': 'Pakistan', 'latitude': 30.0, 'longitude': 70.0}, {'toponym': 'Sindh', 'latitude': 26.13333, 'longitude': 68.76667}, {'toponym': 'Quetta', 'latitude': 30.18414, 'longitude': 67.00141}]\n"
     ]
    }
   ],
   "source": [
    "disaster = \"flood\"\n",
    "country = \"Pakistan\"\n",
    "iso2 = \"PK\"\n",
    "country_iso = 'PAK'\n",
    "month = \"June\"\n",
    "year = 2022\n",
    "location = \"Pakistan\"\n",
    "start_dt = pd.to_datetime(\"2022-06-22\")\n",
    "months_add, weeks_add, days_add = 0,3,0\n",
    "end_dt = start_dt + pd.DateOffset(months=months_add, weeks=weeks_add, days = days_add)\n",
    "start_dt = start_dt.strftime('%Y-%m-%d')\n",
    "end_dt = end_dt.strftime('%Y-%m-%d')\n",
    "print(f\"We are looking specifically from {start_dt} to {end_dt}.\")\n",
    "\n",
    "file_path = '/eos/jeodpp/home/users/mihadar/data/EMM/flood_2022_locations.json'\n",
    "unique_locs = extract_unique_locations(file_path)\n",
    "\n",
    "print(unique_locs)\n",
    "\n",
    "lats = [loc['latitude'] for loc in unique_locs]\n",
    "longs = [loc['longitude'] for loc in unique_locs]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f61ff6ab-7129-4c1c-99c3-8efa12ae143e",
   "metadata": {},
   "outputs": [],
   "source": [
    "json_path=\"/eos/jeodpp/home/users/mihadar/data/ACLED/ACLED_events_by_country.json\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "949f9043-a5a7-4956-bce6-14b8fbe6f230",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading formatted geocoded file...\n",
      "['PK', 'IN', 'AF']\n",
      "['PAK', 'IND', 'AFG']\n"
     ]
    }
   ],
   "source": [
    "relevant_iso2 = relevant_countries(unique_locs, iso2, radius = 200)\n",
    "print(relevant_iso2)\n",
    "relevant_iso3 = convert_iso2_to_iso3(relevant_iso2)\n",
    "print(relevant_iso3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7d5467cc-5c78-4a56-8eda-4a9379369c20",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# matched = fetch_relevant_events(\n",
    "#     json_path=json_path,\n",
    "#     start_date=start_dt,\n",
    "#     iso3_inputs=relevant_iso3,\n",
    "#     past_look_months=2\n",
    "# )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "40f25da6-2b3d-40b0-bf18-956fd56dce69",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Filtering complete. Radius: 50 km.\n",
      "Events before: 899\n",
      "Events after:  112\n",
      "Deleted:       787\n"
     ]
    }
   ],
   "source": [
    "filter_matched_events_acled(\n",
    "    latitudes=lats,\n",
    "    longitudes=longs,\n",
    "    radius=50\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "69ed0d2d-295a-4c63-b50e-99bf7a80fca2",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successfully written 336 ACLED rows to: /eos/jeodpp/home/users/mihadar/data/events/flood_Pakistan_June_2022.csv\n"
     ]
    }
   ],
   "source": [
    "database_path = '/eos/jeodpp/home/users/mihadar/data/events/flood_Pakistan_June_2022.csv'\n",
    "\n",
    "write_acled_to_csv(database_path)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [pin]",
   "language": "python",
   "name": "conda-env-pin-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

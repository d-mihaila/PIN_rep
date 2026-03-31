

app = JRC_PIN_dm2026
limit = 10
disaster = conflict_violence

def ReliefWebAPI(country, disaster, limit):
    """Fetch ReliefWeb articles for a specific country and disaster type."""
    c = country
    d = disaster
    l = limit
    app = CONFIG['reliefweb_appname']

    api_url = f"https://api.reliefweb.int/v1/reports?appname={app}&query[value]={d}&filter[field]=country.iso3&filter[value]={c}&preset=latest&limit={l}"

    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"ReliefWeb API error: {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    output = data['data']
    links = []

    for item in output:
        links.append(item['href'])

    df_articles = pd.DataFrame()

    for link in links:
        article_response = requests.get(link)

        if article_response.status_code == 200:
            article_content = article_response.json()
            article_data = article_content['data']

            for item in article_data:
                if 'fields' in item and 'body' in item['fields']:
                    article_text = str(item['fields']['body'])
                    source_long = item['fields'].get('source', [])
                    source_name = source_long[0].get("shortname") if source_long else "Unknown"
                    date_created = item['fields'].get('date', {}).get('created', '')

                    df = pd.DataFrame({
                        'ISO3' : country,
                        'Disaster': disaster,
                        'URL': link, 
                        'Source': source_name, 
                        'ArticleText': article_text,
                        'Date': date_created
                    }, index=[0])
                    df_articles = pd.concat([df_articles, df], ignore_index=True)

    df_articles_dedup = df_articles.drop_duplicates(subset=['ArticleText']).reset_index(drop=True)

    return df_articles_dedup
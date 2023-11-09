const axios = require('axios');
const fs = require('fs');

(async () => {
    let page = 1
    let pagesCount = 68
    let glossaries = []

    while(page <= pagesCount) {
        const params = {			
            language: 'en-US',
            pageSize: '1000',
            page: page
        }

        const response = await axios.get(`https://help.sap.com/http.svc/glossary`, { params: params })
        glossaries =  [...glossaries, ...response?.data?.data.matches]
        console.log(page)

        page++
    }
    await fs.promises.writeFile(`docs/sap_glossary.json`, JSON.stringify(glossaries, null, 2));
})();
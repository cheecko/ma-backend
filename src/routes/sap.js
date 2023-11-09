const express = require('express')
const dotenv = require('dotenv')
const axios = require('axios')
const fs = require('fs')

const { OpenAI } = require('langchain/llms/openai')
const { PromptTemplate } = require('langchain/prompts')
const { RetrievalQAChain, LLMChain, loadQAStuffChain, loadQAMapReduceChain, APIChain  } = require('langchain/chains')
const { MemoryVectorStore } = require('langchain/vectorstores/memory')
const { OpenAIEmbeddings } = require('langchain/embeddings/openai')
const { Document } = require('langchain/document')
const { PDFLoader } = require('langchain/document_loaders/fs/pdf')
const { JSONLoader } = require('langchain/document_loaders/fs/json')
const { StructuredOutputParser } = require('langchain/output_parsers')
const { ContextualCompressionRetriever } = require('langchain/retrievers/contextual_compression')
const { LLMChainExtractor } = require('langchain/retrievers/document_compressors/chain_extract')
const { CacheBackedEmbeddings } = require('langchain/embeddings/cache_backed')
const { InMemoryStore } = require('langchain/storage/in_memory')

const router = express.Router()
dotenv.config()

// Initialize In Memory Store for Cache Embedding
const inMemoryStore = new InMemoryStore()

const getSAPDocumentation = async (title) => {
	const url = 'https://help.sap.com/http.svc/elasticsearch';

	const params = {
		area: 'content',
		version: '6.18.latest',
		language: 'en-US',
		state: 'PRODUCTION',
		q: title,
		transtype: 'standard,html,pdf,others',
		product: 'SAP_ERP',
		to: 19,
		advancedSearch: 0,
		excludeNotSearchable: 1
	}
	const response = await axios.get(url, { params: params })

	const pages = response?.data?.data?.results.map(page => {
		return {
			// pageContent: `${page.title} - ${page.snippet}`,
			pageContent: `${page.title} - ${page.snippet}`,
			metadata: {term: page.title, url: `https://help.sap.com${page.url}`, source: 'https://help.sap.com/docs/search', search: title}
		}
	})

	return pages
	// return response.data.query.pages
}

//https://js.langchain.com/docs/modules/chains/popular/vector_db_qa - Retrieval QA
router.post('/qa', async (req, res) => {
	try {
		const question = req.body.data.question
		const scenario = req.body.data.scenario ?? 2

		const firstExample = `
		Example 1:
		- Context: The Investment Management (IM) component provides functions for planning, investment, and financing processes for various types of investments such as capital investments, research and development, maintenance programs, and overhead projects.
		- Question: What types of investments does the Investment Management component support?
		- Answer: The Investment Management component supports various types of investments including capital investments, research and development, maintenance programs, and overhead projects.
		- Source: Investment Management (Overview) - Purpose section
		`

		const otherExamples = `
		Example 2:
		- Context: The integration of Asset Accounting (FI-AA) allows for the capitalization of costs from internal orders and work breakdown structure (WBS) elements to fixed assets. Costs not requiring capitalization can be settled to cost accounting.
		- Question: How does Asset Accounting (FI-AA) integrate with Investment Management?
		- Answer: Asset Accounting (FI-AA) integration enables the capitalization of costs from internal orders and WBS elements to fixed assets, along with the settlement of non-capitalizable costs to cost accounting.
		- Source: Investment Management (Overview) - Integration section

		Example 3:
		- Context: Investment Programs serve as a supplement for planning individual measures and budgeting across various areas such as planning, administration, and global budget monitoring. These programs allow for a comprehensive overview of investment planning and strict budget control.
		- Question: What is the purpose of Investment Programs in the context of investment planning?
		- Answer: Investment Programs supplement individual measure planning and budgeting by providing a comprehensive view of investment planning, administration, and global budget monitoring, while maintaining strict budget control.
		- Source: Investment Programs - Purpose section
		`

		// Initialize prompt template to be used to answer the question
		const promptTemplate = `
		Use the following pieces of context to answer the questions at the end.
		The answers should be derived exclusively from the information contained within the retrieved documents.
		Please don't use any your pretrained knowledge to answer the question.
		If you don't have sufficient information from the retrieved context to answer a question, please state that you don't know.

		Please include the source of your answer, indicating the document or page number from which you retrieved the information.
		${scenario == 1 || scenario == 2 ? firstExample : ''}
		${scenario == 2 ? otherExamples : ''}

		- Context: {context}
		- Question: {question}
		- Answer:
		- Source:`;
		const prompt = PromptTemplate.fromTemplate(promptTemplate);

		// Load the PDF documents
		const loader = new PDFLoader('docs/sap_docs.pdf')
		const docs = await loader.load()

		// docs.forEach((doc, index) =>{
		// 	fs.promises.writeFile(`docs/texts/sap_docs_${doc.metadata.loc.pageNumber}.json`, JSON.stringify(doc.pageContent, null, 2))
		// })

		// Initialize the LLM to use to answer the question.
		const embeddings = new OpenAIEmbeddings()
		const model = new OpenAI({
			modelName: 'gpt-3.5-turbo',
			openAIApiKey: process.env.OPENAI_API_KEY,
			temperature: 0,
			callbacks: [
			  {
				handleLLMEnd: (val) => {
					console.log(val)
				},
			  },
			]
		})

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// https://js.langchain.com/docs/modules/data_connection/caching_embeddings - Cache Embedding
		// Initialize Cache Embedding
		cacheBackedEmbeddings = CacheBackedEmbeddings.fromBytesStore(
			embeddings,
			inMemoryStore,
			{ namespace: embeddings.modelName }
		)

		// No keys logged yet since the cache is empty
		let keys1 = [];
		for await (const key of inMemoryStore.yieldKeys()) {
			keys1.push(key);
		}
		// console.log(keys1.slice(0, 5));
		let time = Date.now()

		// Create a vector store from the documents.
		const vectorStore = await MemoryVectorStore.fromDocuments(docs, cacheBackedEmbeddings)
		console.log(`Initial creation time: ${Date.now() - time}ms`)
		const baseRetriever = vectorStore.asRetriever()

		// Many keys logged with hashed values
		let keys2 = [];
		for await (const key of inMemoryStore.yieldKeys()) {
			keys2.push(key);
		}
		// console.log(keys2.slice(0, 5));

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// https://js.langchain.com/docs/modules/data_connection/retrievers/how_to/contextual_compression/ - Contextual Compression
		// Create Contextual Compression to filter irrelevant context
		const baseCompressor = LLMChainExtractor.fromLLM(model)
		
		const retriever = new ContextualCompressionRetriever({
			baseCompressor: baseCompressor,
			baseRetriever: baseRetriever,
		})

		// Create a chain that uses the OpenAI LLM and vector store.
		const chain = new RetrievalQAChain({
			combineDocumentsChain: loadQAStuffChain(model, { prompt }), // loadQAStuffChain or loadQAMapReduceChain
			retriever: baseRetriever, // baseRetriever or retriever
			returnSourceDocuments: true,
			verbose: false
		})

		const response = await chain.call({
			query: question,
		})
		console.log({ response })

		return res.json(response)
	} catch (e) {
		res.status(500).json({ message: 'Something went wrong.' })
		console.error(e)
	}
})

router.post('/el', async (req, res) => {
	try {
		const text = req.body.data.text
		const file = 'docs/sap_glossary_1.json'

		// it's good method, but the structure of metadata is not to be expected
		// const loader = new JSONLoader(file, ['/sterm', '/sglossary', '/scomponent'])
		// const docs = await loader.load()
		// console.log(docs)
		
		const data = JSON.parse(fs.readFileSync(file, 'utf8'));
		const sapGlossary = data.map((doc, index) => {
			return new Document({
				pageContent: `
				${doc.scompdesc} (${doc.scomponent})
				${doc.sterm}
				${doc.sglossary}
				`,
				metadata: {source: file, term: doc.sterm, url: `https://help.sap.com/glossary/?locale=en-US&term=${encodeURIComponent(doc.sterm)}`, line: index}
			})
		})
		// console.log(sapGlossary)

		const embeddings = new OpenAIEmbeddings()
		const model = new OpenAI({
			modelName: 'gpt-3.5-turbo',
			openAIApiKey: process.env.OPENAI_API_KEY,
			temperature: 0
		})

		const parser = StructuredOutputParser.fromNamesAndDescriptions({
			named_entities: "answer to the user's first instruction (array)",
			candidate_entities: "answer to the user's second instruction (object)",
			disambiguation: "answer to the user's third instruction (object)",
		});

		const formatInstructions = parser.getFormatInstructions();

		const template = `
		Given the following text: 
		
		{text}

		1. Identify the named entities mentioned in the text.
		2. Generate a list of candidate entities for each named entity based on the context and rank them in order of relevance.
		3. Disambiguate the candidate entities and select the most suitable matches for each named entity.
		{format_instructions}
		`

		const prompt = new PromptTemplate({
			template: template,
			inputVariables: ["text"],
			partialVariables: { format_instructions: formatInstructions }
		});
	
		const chain = new LLMChain({ llm: model, prompt: prompt, outputKey: 'result', verbose: false });
	
		const response = await chain.call({ text: text });
		console.log(response)
		const json = JSON.parse(response?.result.replace('json', '').replaceAll('`', ''))
		console.log(json)

		const sapDocs = await Promise.all(json.named_entities.map(async entity => {
			let wikiDocs = await getSAPDocumentation(entity) 
			if(wikiDocs.length == 0) wikiDocs = await getSAPDocumentation(entity)
			return wikiDocs
		}))

		const docs = [...sapGlossary, ...sapDocs.flat(1)]
		// const docs = [...sapGlossary]
	
		const store = await MemoryVectorStore.fromDocuments(docs, embeddings);

		const linking = await Promise.all(json.named_entities.map(async entity => {
			entity = entity.trim()
	
			const relevantDocs = await store.similaritySearchWithScore(`${entity} - ${json.disambiguation[entity]}`, 1);
			console.log(`${entity} - ${json.disambiguation[entity]}`)
			console.log(relevantDocs)
	
			// const relevantDocs = await store.similaritySearch(json.disambiguation[entity], 1);
			// console.log(entity)
			// console.log(relevantDocs)
			return { entity: entity, document: relevantDocs[0] }
		}))

		console.log(linking)
		return res.json({ ...json, entity_linking: linking, sourceDocuments: docs })
	} catch (e) {
		res.status(500).json({ message: 'Something went wrong.' })
		console.error(e)
	}
})

module.exports = router

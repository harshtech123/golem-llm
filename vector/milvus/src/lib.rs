use crate::client::MilvusClient;
use crate::conversions::{
    collection_info_to_export_collection_info,
    vector_records_to_upsert_request, create_search_request, create_query_request,
    create_get_request, create_delete_request, milvus_search_results_to_search_results,
    milvus_entities_to_vector_records, collection_stats_to_export_stats,
    milvus_error_to_vector_error, distance_metric_to_string,
};
use golem_vector::config::{with_config_key, get_optional_config, with_connection_config_key};
use golem_vector::durability::ExtendedGuest;
use golem_vector::exports::golem::vector::{
    analytics::Guest as AnalyticsGuest,
    collections::{Guest as CollectionsGuest, CollectionInfo},
    connection::{Credentials, Guest as ConnectionGuest},
    namespaces::Guest as NamespacesGuest,
    search::Guest as SearchGuest,
    search_extended::Guest as SearchExtendedGuest,
    types::{
        DistanceMetric, FilterExpression, Id, Metadata, SearchResult, VectorData,
        VectorError, VectorRecord,
    },
    vectors::Guest as VectorsGuest,
};

mod client;
mod conversions;

struct MilvusComponent;

impl MilvusComponent {
    const URI_ENV_VAR: &'static str = "MILVUS_URI";
    const TOKEN_ENV_VAR: &'static str = "MILVUS_TOKEN";
    const DATABASE_ENV_VAR: &'static str = "MILVUS_DATABASE";

    fn create_client() -> Result<MilvusClient, VectorError> {
        let uri = with_config_key(
            Self::URI_ENV_VAR,
            |e| Err(VectorError::ConnectionError(format!("Missing URI: {e}"))),
            |value| Ok(value),
        ).unwrap_or_else(|_| "http://localhost:19530".to_string());

        let token = get_optional_config(Self::TOKEN_ENV_VAR);
        let database = get_optional_config(Self::DATABASE_ENV_VAR);

        Ok(MilvusClient::new(uri, token, database))
    }

    fn create_client_with_options(options: &Option<Metadata>) -> Result<MilvusClient, VectorError> {
        let uri = with_connection_config_key(options, "uri")
            .or_else(|| get_optional_config(Self::URI_ENV_VAR))
            .unwrap_or_else(|| "http://localhost:19530".to_string());

        let token = with_connection_config_key(options, "token")
            .or_else(|| get_optional_config(Self::TOKEN_ENV_VAR));

        let database = with_connection_config_key(options, "database")
            .or_else(|| get_optional_config(Self::DATABASE_ENV_VAR));

        Ok(MilvusClient::new(uri, token, database))
    }
}

impl ExtendedGuest for MilvusComponent {
    fn connect_internal(
        _endpoint: &str,
        _credentials: &Option<Credentials>,
        _timeout_ms: &Option<u32>,
        options: &Option<Metadata>,
    ) -> Result<(), VectorError> {
        let _client = Self::create_client_with_options(options)?;
        Ok(())
    }
}

impl ConnectionGuest for MilvusComponent {
    fn connect(
        _endpoint: String,
        _credentials: Option<Credentials>,
        _timeout_ms: Option<u32>,
        options: Option<Metadata>,
    ) -> Result<(), VectorError> {
        let _client = Self::create_client_with_options(&options)?;
        Ok(())
    }

    fn disconnect() -> Result<(), VectorError> {
        Ok(())
    }

    fn get_connection_status() -> Result<golem_vector::exports::golem::vector::connection::ConnectionStatus, VectorError> {
        match Self::create_client() {
            Ok(client) => {
                match client.list_collections() {
                    Ok(_) => Ok(golem_vector::exports::golem::vector::connection::ConnectionStatus {
                        connected: true,
                        provider: Some("milvus".to_string()),
                        endpoint: Some(client.base_url().to_string()),
                        last_activity: None,
                        connection_id: Some("milvus-api".to_string()),
                    }),
                    Err(_) => Ok(golem_vector::exports::golem::vector::connection::ConnectionStatus {
                        connected: false,
                        provider: Some("milvus".to_string()),
                        endpoint: Some(client.base_url().to_string()),
                        last_activity: None,
                        connection_id: Some("milvus-api".to_string()),
                    }),
                }
            }
            Err(_) => Ok(golem_vector::exports::golem::vector::connection::ConnectionStatus {
                connected: false,
                provider: Some("milvus".to_string()),
                endpoint: Some("http://localhost:19530".to_string()),
                last_activity: None,
                connection_id: Some("milvus-api".to_string()),
            }),
        }
    }

    fn test_connection(
        _endpoint: String,
        _credentials: Option<Credentials>,
        _timeout_ms: Option<u32>,
        options: Option<Metadata>,
    ) -> Result<bool, VectorError> {
        match Self::create_client_with_options(&options) {
            Ok(client) => {
                match client.list_collections() {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Ok(false),
        }
    }
}

impl CollectionsGuest for MilvusComponent {
    fn upsert_collection(
        name: String,
        description: Option<String>,
        dimension: u32,
        metric: DistanceMetric,
        _index_config: Option<golem_vector::exports::golem::vector::collections::IndexConfig>,
        _metadata: Option<Metadata>,
    ) -> Result<CollectionInfo, VectorError> {
        let client = Self::create_client()?;
        
        let create_request = client::CreateCollectionRequest {
            db_name: client.database().to_string(),
            collection_name: name.clone(),
            dimension,
            metric_type: Some(distance_metric_to_string(&metric)),
            primary_field: Some("id".to_string()),
            vector_field: Some("vector".to_string()),
            description,
            enable_dynamic_field: Some(true),
            schema: None,
            index_params: None,
        };

        match client.create_collection(&create_request) {
            Ok(_) => {
                match client.load_collection(&name) {
                    Ok(_) => {
                        match client.describe_collection(&name) {
                            Ok(response) => collection_info_to_export_collection_info(&response.data),
                            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
                        }
                    }
                    Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn list_collections() -> Result<Vec<String>, VectorError> {
        let client = Self::create_client()?;
        
        match client.list_collections() {
            Ok(response) => {
                if response.code == 0 {
                    Ok(response.data)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn get_collection(name: String) -> Result<CollectionInfo, VectorError> {
        let client = Self::create_client()?;
        
        match client.describe_collection(&name) {
            Ok(response) => {
                if response.code == 0 {
                    collection_info_to_export_collection_info(&response.data)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn update_collection(
        name: String,
        _description: Option<String>,
        _metadata: Option<Metadata>,
    ) -> Result<CollectionInfo, VectorError> {
        Self::get_collection(name)
    }

    fn delete_collection(name: String) -> Result<(), VectorError> {
        let client = Self::create_client()?;
        
        let _ = client.release_collection(&name);
        
        match client.drop_collection(&name) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(())
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn collection_exists(name: String) -> Result<bool, VectorError> {
        let client = Self::create_client()?;
        
        match client.has_collection(&name) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(response.data.has)
                } else {
                    Ok(false)
                }
            }
            Err(_) => Ok(false),
        }
    }
}

impl VectorsGuest for MilvusComponent {
    fn upsert_vectors(
        collection: String,
        vectors: Vec<VectorRecord>,
        _namespace: Option<String>,
    ) -> Result<golem_vector::exports::golem::vector::vectors::BatchResult, VectorError> {
        let client = Self::create_client()?;
        
        let upsert_request = vector_records_to_upsert_request(&collection, client.database(), &vectors)?;
        
        match client.upsert(&upsert_request) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(golem_vector::exports::golem::vector::vectors::BatchResult {
                        success_count: response.data.upsert_count,
                        failure_count: 0,
                        errors: vec![],
                    })
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn upsert_vector(
        collection: String,
        id: Id,
        vector: VectorData,
        metadata: Option<Metadata>,
        namespace: Option<String>,
    ) -> Result<(), VectorError> {
        let record = VectorRecord {
            id,
            vector,
            metadata,
        };
        
        let result = Self::upsert_vectors(collection, vec![record], namespace)?;
        
        if result.success_count > 0 {
            Ok(())
        } else {
            Err(VectorError::ProviderError("Failed to upsert vector".to_string()))
        }
    }

    fn get_vectors(
        collection: String,
        ids: Vec<Id>,
        _namespace: Option<String>,
        include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<VectorRecord>, VectorError> {
        let client = Self::create_client()?;
        
        let mut output_fields = Vec::new();
        if include_vectors.unwrap_or(true) {
            output_fields.push("vector".to_string());
        }

        let get_request = create_get_request(
            &collection,
            client.database(),
            &ids,
            if output_fields.is_empty() { None } else { Some(&output_fields) },
        );
        
        match client.get(&get_request) {
            Ok(response) => {
                if response.code == 0 {
                    milvus_entities_to_vector_records(&response.data)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn get_vector(
        collection: String,
        id: Id,
        namespace: Option<String>,
    ) -> Result<Option<VectorRecord>, VectorError> {
        let vectors = Self::get_vectors(collection, vec![id], namespace, Some(true), Some(true))?;
        Ok(vectors.into_iter().next())
    }

    fn update_vector(
        collection: String,
        id: Id,
        vector: Option<VectorData>,
        metadata: Option<Metadata>,
        namespace: Option<String>,
        _merge_metadata: Option<bool>,
    ) -> Result<(), VectorError> {
        if let Some(vector_data) = vector {
            Self::upsert_vector(collection, id, vector_data, metadata, namespace)
        } else {
            Err(VectorError::InvalidParams("Vector data is required for update".to_string()))
        }
    }

    fn delete_vectors(
        collection: String,
        ids: Vec<Id>,
        _namespace: Option<String>,
    ) -> Result<u32, VectorError> {
        let client = Self::create_client()?;
        
        let delete_request = create_delete_request(
            &collection,
            client.database(),
            Some(&ids),
            None,
        )?;
        
        match client.delete(&delete_request) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(response.data.delete_count)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn delete_by_filter(
        collection: String,
        filter: FilterExpression,
        _namespace: Option<String>,
    ) -> Result<u32, VectorError> {
        let client = Self::create_client()?;
        
        let delete_request = create_delete_request(
            &collection,
            client.database(),
            None,
            Some(&filter),
        )?;
        
        match client.delete(&delete_request) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(response.data.delete_count)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn delete_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<u32, VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces like Pinecone".to_string()))
    }

    fn list_vectors(
        collection: String,
        _namespace: Option<String>,
        filter: Option<FilterExpression>,
        limit: Option<u32>,
        _cursor: Option<String>,
        include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<golem_vector::exports::golem::vector::vectors::ListResponse, VectorError> {
        let client = Self::create_client()?;
        
        let mut output_fields = vec!["id".to_string()];
        if include_vectors.unwrap_or(false) {
            output_fields.push("vector".to_string());
        }
        
        let query_request = create_query_request(
            &collection,
            client.database(),
            None,
            filter.as_ref(),
            if output_fields.len() == 1 { None } else { Some(&output_fields) },
            limit,
            None,
        )?;
        
        match client.query(&query_request) {
            Ok(response) => {
                if response.code == 0 {
                    let vector_records = milvus_entities_to_vector_records(&response.data)?;
                    
                    Ok(golem_vector::exports::golem::vector::vectors::ListResponse {
                        vectors: vector_records,
                        next_cursor: None, 
                        total_count: None,
                    })
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn count_vectors(
        collection: String,
        filter: Option<FilterExpression>,
        _namespace: Option<String>,
    ) -> Result<u64, VectorError> {
        let client = Self::create_client()?;
        
        if filter.is_some() {
            let query_request = create_query_request(
                &collection,
                client.database(),
                None,
                filter.as_ref(),
                Some(&vec!["id".to_string()]),
                None,
                None,
            )?;
            
            match client.query(&query_request) {
                Ok(response) => {
                    if response.code == 0 {
                        Ok(response.data.len() as u64)
                    } else {
                        Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                    }
                }
                Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
            }
        } else {
            match client.get_collection_stats(&collection) {
                Ok(response) => {
                    if response.code == 0 {
                        Ok(response.data.row_count)
                    } else {
                        Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                    }
                }
                Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
            }
        }
    }
}

impl SearchGuest for MilvusComponent {
    fn search_vectors(
        collection: String,
        query: golem_vector::exports::golem::vector::search::SearchQuery,
        limit: u32,
        filter: Option<FilterExpression>,
        _namespace: Option<String>,
        include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
        min_score: Option<f32>,
        max_distance: Option<f32>,
        _search_params: Option<Vec<(String, String)>>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        let client = Self::create_client()?;
        
        let mut output_fields = vec!["id".to_string()];
        if include_vectors.unwrap_or(false) {
            output_fields.push("vector".to_string());
        }
        
        let search_request = create_search_request(
            &collection,
            client.database(),
            &query,
            limit,
            filter.as_ref(),
            if output_fields.len() == 1 { None } else { Some(&output_fields) },
            "vector",
            "COSINE", 
        )?;
        
        match client.search(&search_request) {
            Ok(response) => {
                if response.code == 0 {
                    let mut results = milvus_search_results_to_search_results(&response.data)?;
                    
                    if let Some(min_score_val) = min_score {
                        results.retain(|result| result.score >= min_score_val);
                    }
                    
                    if let Some(max_distance_val) = max_distance {
                        results.retain(|result| result.distance <= max_distance_val);
                    }
                    
                    Ok(results)
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn find_similar(
        collection: String,
        vector: VectorData,
        limit: u32,
        namespace: Option<String>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        use golem_vector::exports::golem::vector::search::SearchQuery;
        
        Self::search_vectors(
            collection,
            SearchQuery::Vector(vector),
            limit,
            None,
            namespace,
            Some(false),
            Some(false),
            None,
            None,
            None,
        )
    }

    fn batch_search(
        collection: String,
        queries: Vec<golem_vector::exports::golem::vector::search::SearchQuery>,
        limit: u32,
        filter: Option<FilterExpression>,
        namespace: Option<String>,
        include_vectors: Option<bool>,
        include_metadata: Option<bool>,
        search_params: Option<Vec<(String, String)>>,
    ) -> Result<Vec<Vec<SearchResult>>, VectorError> {
        let mut results = Vec::new();
        
        for query in queries {
            let result = Self::search_vectors(
                collection.clone(),
                query,
                limit,
                filter.clone(),
                namespace.clone(),
                include_vectors,
                include_metadata,
                None,
                None,
                search_params.clone(),
            )?;
            results.push(result);
        }
        
        Ok(results)
    }
}

impl SearchExtendedGuest for MilvusComponent {
    fn recommend_vectors(
        _collection: String,
        _positive: Vec<golem_vector::exports::golem::vector::search_extended::RecommendationExample>,
        _negative: Option<Vec<golem_vector::exports::golem::vector::search_extended::RecommendationExample>>,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _strategy: Option<golem_vector::exports::golem::vector::search_extended::RecommendationStrategy>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature("Recommendation search not supported by Milvus".to_string()))
    }

    fn discover_vectors(
        _collection: String,
        _context_pairs: Vec<golem_vector::exports::golem::vector::search_extended::ContextPair>,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature("Discovery search not supported by Milvus".to_string()))
    }

    fn search_groups(
        _collection: String,
        _query: golem_vector::exports::golem::vector::search::SearchQuery,
        _group_by: String,
        _group_size: u32,
        _max_groups: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<golem_vector::exports::golem::vector::search_extended::GroupedSearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature("Grouped search not supported by Milvus".to_string()))
    }

    fn search_range(
        _collection: String,
        _vector: VectorData,
        _min_distance: Option<f32>,
        _max_distance: f32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _limit: Option<u32>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature("Range search not supported by Milvus".to_string()))
    }

    fn search_text(
        _collection: String,
        _query_text: String,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature("Text search not supported by Milvus".to_string()))
    }
}

impl AnalyticsGuest for MilvusComponent {
    fn get_collection_stats(
        collection: String,
        _namespace: Option<String>,
    ) -> Result<golem_vector::exports::golem::vector::analytics::CollectionStats, VectorError> {
        let client = Self::create_client()?;
        
        match client.get_collection_stats(&collection) {
            Ok(response) => {
                if response.code == 0 {
                    Ok(collection_stats_to_export_stats(&response.data))
                } else {
                    Err(VectorError::ProviderError(format!("Milvus error code: {}", response.code)))
                }
            }
            Err(e) => Err(milvus_error_to_vector_error(&e.to_string())),
        }
    }

    fn get_field_stats(
        _collection: String,
        _field: String,
        _namespace: Option<String>,
    ) -> Result<golem_vector::exports::golem::vector::analytics::FieldStats, VectorError> {
        Err(VectorError::UnsupportedFeature("Field stats not supported by Milvus".to_string()))
    }

    fn get_field_distribution(
        _collection: String,
        _field: String,
        _limit: Option<u32>,
        _namespace: Option<String>,
    ) -> Result<Vec<(golem_vector::exports::golem::vector::types::MetadataValue, u64)>, VectorError> {
        Err(VectorError::UnsupportedFeature("Field distribution not supported by Milvus".to_string()))
    }
}

impl NamespacesGuest for MilvusComponent {
    fn upsert_namespace(
        _collection: String,
        _namespace: String,
        _metadata: Option<Metadata>,
    ) -> Result<golem_vector::exports::golem::vector::namespaces::NamespaceInfo, VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces".to_string()))
    }

    fn list_namespaces(
        _collection: String,
    ) -> Result<Vec<golem_vector::exports::golem::vector::namespaces::NamespaceInfo>, VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces".to_string()))
    }

    fn get_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<golem_vector::exports::golem::vector::namespaces::NamespaceInfo, VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces".to_string()))
    }

    fn delete_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<(), VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces".to_string()))
    }

    fn namespace_exists(
        _collection: String,
        _namespace: String,
    ) -> Result<bool, VectorError> {
        Err(VectorError::UnsupportedFeature("Milvus doesn't support namespaces".to_string()))
    }
}

impl golem_vector::exports::golem::vector::types::Guest for MilvusComponent {
    type MetadataFunc = golem_vector::exports::golem::vector::types::MetadataValue;
    type FilterFunc = golem_vector::exports::golem::vector::types::FilterExpression;
}

golem_vector::export_vector!(MilvusComponent with_types_in golem_vector);

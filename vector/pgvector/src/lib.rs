use crate::client::PgVectorClient;
use crate::conversions::{
    table_info_to_export_collection_info,
    vector_records_to_pgvector_data, create_table_request_from_collection_info,
    create_search_request, pg_search_results_to_search_results,
    pg_vector_results_to_vector_records, count_response_to_export_stats, 
};
use golem_vector::config::{with_config_key, get_optional_config, with_connection_config_key};
use golem_vector::durability::{ExtendedGuest, DurableVector};
use golem_vector::golem::vector::{
    analytics::{Guest as AnalyticsGuest, FieldStats, CollectionStats},
    collections::{Guest as CollectionsGuest, CollectionInfo, IndexConfig},
    connection::{Credentials, Guest as ConnectionGuest, ConnectionStatus},
    namespaces::{Guest as NamespacesGuest, NamespaceInfo},
    search::{Guest as SearchGuest, SearchQuery},
    search_extended::{Guest as SearchExtendedGuest, GroupedSearchResult, RecommendationExample, RecommendationStrategy, ContextPair},
    types::{
        DistanceMetric, FilterExpression, Id, Metadata, SearchResult, VectorData,
        VectorError, VectorRecord, MetadataValue,
    },
    vectors::{Guest as VectorsGuest, ListResponse, BatchResult},
};

mod client;
mod conversions;

struct PgVectorComponent;

impl PgVectorComponent {
    const BASE_URL_ENV_VAR: &'static str = "PGVECTOR_BASE_URL";
    const API_KEY_ENV_VAR: &'static str = "PGVECTOR_API_KEY";

    fn create_client() -> Result<PgVectorClient, VectorError> {
        let base_url = with_config_key(
            Self::BASE_URL_ENV_VAR,
            |e| Err(VectorError::ConnectionError(format!("Missing base URL: {e}"))),
            |value| Ok(value),
        ).unwrap_or_else(|_| "http://localhost:3000".to_string());

        let api_key = get_optional_config(Self::API_KEY_ENV_VAR);

        Ok(PgVectorClient::new(base_url, api_key))
    }

    fn create_client_with_options(options: &Option<Metadata>) -> Result<PgVectorClient, VectorError> {
        let base_url = with_connection_config_key(options, "base_url")
            .unwrap_or_else(|| "http://localhost:3000".to_string());

        let api_key = with_connection_config_key(options, "api_key");

        Ok(PgVectorClient::new(base_url, api_key))
    }
}

impl ExtendedGuest for PgVectorComponent {
    fn connect_internal(
        _endpoint: &str,
        _credentials: &Option<Credentials>,
        _timeout_ms: &Option<u32>,
        options: &Option<Metadata>,
    ) -> Result<(), VectorError> {
        let client = Self::create_client_with_options(options)?;
        client.enable_extension()?;
        Ok(())
    }
}

impl ConnectionGuest for PgVectorComponent {
    fn connect(
        _endpoint: String,
        _credentials: Option<Credentials>,
        _timeout_ms: Option<u32>,
        options: Option<Metadata>,
    ) -> Result<(), VectorError> {
        let client = Self::create_client_with_options(&options)?;
        client.enable_extension()?;
        Ok(())
    }

    fn disconnect() -> Result<(), VectorError> {
        Ok(())
    }

    fn get_connection_status() -> Result<ConnectionStatus, VectorError> {
        match Self::create_client() {
            Ok(client) => {
                match client.enable_extension() {
                    Ok(_) => Ok(ConnectionStatus {
                        connected: true,
                        provider: Some("pgvector".to_string()),
                        endpoint: Some(client.base_url().to_string()),
                        last_activity: None,
                        connection_id: Some("pgvector-http".to_string()),
                    }),
                    Err(_) => Ok(ConnectionStatus {
                        connected: false,
                        provider: Some("pgvector".to_string()),
                        endpoint: Some(client.base_url().to_string()),
                        last_activity: None,
                        connection_id: Some("pgvector-http".to_string()),
                    }),
                }
            }
            Err(_) => Ok(ConnectionStatus {
                connected: false,
                provider: Some("pgvector".to_string()),
                endpoint: None,
                last_activity: None,
                connection_id: Some("pgvector-http".to_string()),
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
                match client.enable_extension() {
                    Ok(_) => Ok(true),
                    Err(_) => Ok(false),
                }
            }
            Err(_) => Ok(false),
        }
    }
}

impl CollectionsGuest for PgVectorComponent {
    fn upsert_collection(
        name: String,
        _description: Option<String>,
        dimension: u32,
        _metric: DistanceMetric,
        _index_config: Option<IndexConfig>,
        metadata: Option<Metadata>,
    ) -> Result<CollectionInfo, VectorError> {
        let client = Self::create_client()?;
        
        client.enable_extension()?;
        
        let create_request = create_table_request_from_collection_info(
            name.clone(),
            dimension,
            metadata.as_ref()
        );

        match client.create_table(&create_request) {
            Ok(_) => {
                let describe_response = client.describe_table(&name)?;
                let count_response = client.count_vectors(&name)?;
                
                table_info_to_export_collection_info(
                    &name,
                    &describe_response.columns,
                    count_response.count
                )
            }
            Err(e) => Err(e),
        }
    }

    fn list_collections() -> Result<Vec<String>, VectorError> {
        let client = Self::create_client()?;
        
        match client.list_tables() {
            Ok(response) => Ok(response.tables),
            Err(e) => Err(e),
        }
    }

    fn get_collection(name: String) -> Result<CollectionInfo, VectorError> {
        let client = Self::create_client()?;
        
        let describe_response = client.describe_table(&name)?;
        let count_response = client.count_vectors(&name)?;
        
        table_info_to_export_collection_info(
            &name,
            &describe_response.columns,
            count_response.count
        )
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
        
        match client.drop_table(&name) {
            Ok(_) => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn collection_exists(name: String) -> Result<bool, VectorError> {
        let client = Self::create_client()?;
        
        match client.table_exists(&name) {
            Ok(response) => Ok(response.exists),
            Err(e) => Err(e),
        }
    }
}

impl VectorsGuest for PgVectorComponent {
    fn upsert_vectors(
        collection: String,
        vectors: Vec<VectorRecord>,
        _namespace: Option<String>,
    ) -> Result<BatchResult, VectorError> {
        let client = Self::create_client()?;
        
        let pg_vectors = vector_records_to_pgvector_data(&vectors)?;
        
        let upsert_request = client::UpsertVectorsRequest {
            table_name: collection,
            vectors: pg_vectors,
        };
        
        match client.upsert_vectors(&upsert_request) {
            Ok(response) => Ok(BatchResult {
                success_count: response.upserted_count,
                failure_count: vectors.len() as u32 - response.upserted_count,
                errors: Vec::new(),
            }),
            Err(e) => Err(e),
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
        include_metadata: Option<bool>,
    ) -> Result<Vec<VectorRecord>, VectorError> {
        let client = Self::create_client()?;
        
        let mut select_columns = vec!["id".to_string()];
        if include_vectors.unwrap_or(true) {
            select_columns.push("embedding".to_string());
        }
        
        let describe_response = client.describe_table(&collection)?;
        if include_metadata.unwrap_or(true) {
            for column in &describe_response.columns {
                if column.name != "id" && column.name != "embedding" {
                    select_columns.push(column.name.clone());
                }
            }
        }

        let get_request = client::GetVectorsRequest {
            table_name: collection,
            ids,
            select_columns,
        };
        
        match client.get_vectors(&get_request) {
            Ok(response) => Ok(pg_vector_results_to_vector_records(&response.results)),
            Err(e) => Err(e),
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
        
        let delete_request = client::DeleteVectorsRequest {
            table_name: collection,
            ids,
        };
        
        match client.delete_vectors(&delete_request) {
            Ok(response) => Ok(response.deleted_count),
            Err(e) => Err(e),
        }
    }

    fn delete_by_filter(
        _collection: String,
        _filter: FilterExpression,
        _namespace: Option<String>,
    ) -> Result<u32, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Delete by filter not yet implemented for pgvector".to_string()
        ))
    }

    fn delete_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<u32, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }

    fn list_vectors(
        _collection: String,
        _namespace: Option<String>,
        _filter: Option<FilterExpression>,
        _limit: Option<u32>,
        _cursor: Option<String>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<ListResponse, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "List vectors not yet implemented for pgvector".to_string()
        ))
    }

    fn count_vectors(
        collection: String,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
    ) -> Result<u64, VectorError> {
        let client = Self::create_client()?;
        
        match client.count_vectors(&collection) {
            Ok(response) => Ok(response.count),
            Err(e) => Err(e),
        }
    }
}

impl SearchGuest for PgVectorComponent {
    fn search_vectors(
        collection: String,
        query: SearchQuery,
        limit: u32,
        filter: Option<FilterExpression>,
        _namespace: Option<String>,
        include_vectors: Option<bool>,
        include_metadata: Option<bool>,
        _min_score: Option<f32>,
        _max_distance: Option<f32>,
        search_params: Option<Vec<(String, String)>>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        let client = Self::create_client()?;
        
        let mut output_fields = vec!["id".to_string()];
        if include_vectors.unwrap_or(false) {
            output_fields.push("embedding".to_string());
        }
        
        if include_metadata.unwrap_or(true) {
            let describe_response = client.describe_table(&collection)?;
            for column in &describe_response.columns {
                if column.name != "id" && column.name != "embedding" {
                    output_fields.push(column.name.clone());
                }
            }
        }
        
        let distance_metric = search_params
            .as_ref()
            .and_then(|params| {
                params.iter().find(|(k, _)| k == "metric").map(|(_, v)| v.clone())
            })
            .unwrap_or_else(|| "cosine".to_string());
        
        let search_request = create_search_request(
            &collection,
            &query,
            limit,
            filter.as_ref(),
            Some(&output_fields),
            &distance_metric,
        )?;
        
        match client.search_vectors(&search_request) {
            Ok(response) => Ok(pg_search_results_to_search_results(&response.results)),
            Err(e) => Err(e),
        }
    }

    fn find_similar(
        collection: String,
        vector: VectorData,
        limit: u32,
        namespace: Option<String>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Self::search_vectors(
            collection,
            SearchQuery::Vector(vector),
            limit,
            None,
            namespace,
            Some(false),
            Some(true),
            None,
            None,
            Some(vec![("metric".to_string(), "cosine".to_string())]),
        )
    }

    fn batch_search(
        collection: String,
        queries: Vec<SearchQuery>,
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

impl SearchExtendedGuest for PgVectorComponent {
    fn recommend_vectors(
        _collection: String,
        _positive: Vec<RecommendationExample>,
        _negative: Option<Vec<RecommendationExample>>,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _strategy: Option<RecommendationStrategy>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Recommendation search not supported by pgvector".to_string()
        ))
    }

    fn discover_vectors(
        _collection: String,
        _target: Option<RecommendationExample>,
        _context: Vec<ContextPair>,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Discovery search not supported by pgvector".to_string()
        ))
    }

    fn search_groups(
        _collection: String,
        _query: SearchQuery,
        _group_by: String,
        _limit: u32,
        _group_limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
        _include_vectors: Option<bool>,
        _include_metadata: Option<bool>,
    ) -> Result<Vec<GroupedSearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Group search not supported by pgvector".to_string()
        ))
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
        Err(VectorError::UnsupportedFeature(
            "Range search not yet implemented for pgvector".to_string()
        ))
    }

    fn search_text(
        _collection: String,
        _query_text: String,
        _limit: u32,
        _filter: Option<FilterExpression>,
        _namespace: Option<String>,
    ) -> Result<Vec<SearchResult>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Text search not supported by pgvector (requires text embedding model)".to_string()
        ))
    }
}

impl AnalyticsGuest for PgVectorComponent {
    fn get_collection_stats(
        collection: String,
        _namespace: Option<String>,
    ) -> Result<CollectionStats, VectorError> {
        let client = Self::create_client()?;
        
        let count_response = client.count_vectors(&collection)?;
        let describe_response = client.describe_table(&collection)?;
        
        let dimension = describe_response.columns.iter()
            .find(|col| col.data_type.starts_with("vector"))
            .and_then(|col| {
                if col.data_type.starts_with("vector(") && col.data_type.ends_with(')') {
                    let dim_str = &col.data_type[7..col.data_type.len()-1];
                    dim_str.parse::<u32>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);
        
        Ok(count_response_to_export_stats(&count_response, dimension))
    }

    fn get_field_stats(
        _collection: String,
        _field: String,
        _namespace: Option<String>,
    ) -> Result<FieldStats, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Field statistics not yet implemented for pgvector".to_string()
        ))
    }

    fn get_field_distribution(
        _collection: String,
        _field: String,
        _limit: Option<u32>,
        _namespace: Option<String>,
    ) -> Result<Vec<(MetadataValue, u64)>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "Field distribution not yet implemented for pgvector".to_string()
        ))
    }
}

impl NamespacesGuest for PgVectorComponent {
    fn upsert_namespace(
        _collection: String,
        _namespace: String,
        _metadata: Option<Metadata>,
    ) -> Result<NamespaceInfo, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }

    fn list_namespaces(
        _collection: String,
    ) -> Result<Vec<NamespaceInfo>, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }

    fn get_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<NamespaceInfo, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }

    fn delete_namespace(
        _collection: String,
        _namespace: String,
    ) -> Result<(), VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }

    fn namespace_exists(
        _collection: String,
        _namespace: String,
    ) -> Result<bool, VectorError> {
        Err(VectorError::UnsupportedFeature(
            "PostgreSQL doesn't support namespaces like Pinecone".to_string()
        ))
    }
}

type DurablePgVectorComponent = DurableVector<PgVectorComponent>;

golem_vector::export_vector!(DurablePgVectorComponent with_types_in golem_vector);

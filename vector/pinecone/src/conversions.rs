use crate::client::{
    IndexModel, QueryResponse, ScoredVector, UpsertRequest, Vector,
};
use golem_vector::golem::vector::types::{
    DistanceMetric, Metadata, MetadataValue, SearchResult, VectorData, VectorError,
    VectorRecord, FilterExpression
};
use golem_vector::golem::vector::search::SearchQuery;
use golem_vector::golem::vector::collections::CollectionInfo;
use serde_json::json;
use std::collections::HashMap;

pub fn vector_data_to_dense(vector_data: &VectorData) -> Vec<f32> {
    match vector_data {
        VectorData::Dense(dense) => dense.clone(),
        VectorData::Sparse(sparse) => {
            let mut dense = vec![0.0; sparse.total_dimensions as usize];
            for (i, &index) in sparse.indices.iter().enumerate() {
                if let Some(value) = sparse.values.get(i) {
                    if (index as usize) < dense.len() {
                        dense[index as usize] = *value;
                    }
                }
            }
            dense
        }
        VectorData::Binary(_) => {
            vec![]
        }
        VectorData::Half(half) => half.data.clone(),
        VectorData::Named(named) => {
            named
                .first()
                .map(|(_, vector)| vector.clone())
                .unwrap_or_default()
        }
        VectorData::Hybrid((dense, _sparse)) => dense.clone(),
    }
}

pub fn dense_to_vector_data(dense: Vec<f32>) -> VectorData {
    VectorData::Dense(dense)
}

pub fn vector_data_to_sparse(vector_data: &VectorData) -> Option<crate::client::SparseValues> {
    match vector_data {
        VectorData::Sparse(sparse) => {
            Some(crate::client::SparseValues {
                indices: sparse.indices.clone(),
                values: sparse.values.clone(),
            })
        }
        VectorData::Hybrid((_dense, sparse)) => {
            Some(crate::client::SparseValues {
                indices: sparse.indices.clone(),
                values: sparse.values.clone(),
            })
        }
        VectorData::Dense(dense) => {
            let mut indices = Vec::new();
            let mut values = Vec::new();
            
            for (i, &value) in dense.iter().enumerate() {
                if value != 0.0 {
                    indices.push(i as u32);
                    values.push(value);
                }
            }
            
            if indices.is_empty() {
                None
            } else {
                Some(crate::client::SparseValues { indices, values })
            }
        }
        _ => None,
    }
}

// metadata conversion

pub fn metadata_to_json_map(
    metadata: &Metadata,
) -> Result<HashMap<String, serde_json::Value>, VectorError> {
    let mut map = HashMap::new();
    for (key, value) in metadata {
        let json_value = match value {
            MetadataValue::StringVal(s) => serde_json::Value::String(s.clone()),
            MetadataValue::NumberVal(n) => serde_json::Value::Number(
                serde_json::Number::from_f64(*n).ok_or_else(|| {
                    VectorError::InvalidParams("Invalid number value".to_string())
                })?,
            ),
            MetadataValue::IntegerVal(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            MetadataValue::BooleanVal(b) => serde_json::Value::Bool(*b),
            MetadataValue::NullVal => serde_json::Value::Null,
            _ => {
                return Err(VectorError::UnsupportedFeature(
                    "Complex metadata types not supported in Pinecone".to_string(),
                ));
            }
        };
        map.insert(key.clone(), json_value);
    }
    Ok(map)
}

pub fn json_map_to_metadata(
    map: &HashMap<String, serde_json::Value>,
) -> Result<Metadata, VectorError> {
    let mut metadata = Vec::new();
    for (key, value) in map {
        let metadata_value = match value {
            serde_json::Value::String(s) => MetadataValue::StringVal(s.clone()),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    MetadataValue::IntegerVal(i)
                } else if let Some(f) = n.as_f64() {
                    MetadataValue::NumberVal(f)
                } else {
                    MetadataValue::NullVal
                }
            }
            serde_json::Value::Bool(b) => MetadataValue::BooleanVal(*b),
            serde_json::Value::Null => MetadataValue::NullVal,
            _ => MetadataValue::NullVal,
        };
        metadata.push((key.clone(), metadata_value));
    }
    Ok(metadata)
}

pub fn distance_metric_to_string(metric: &DistanceMetric) -> String {
    match metric {
        DistanceMetric::Cosine => "cosine".to_string(),
        DistanceMetric::Euclidean => "euclidean".to_string(),
        DistanceMetric::DotProduct => "dotproduct".to_string(),
        _ => "cosine".to_string(), 
    }
}

pub fn string_to_distance_metric(metric: &str) -> DistanceMetric {
    match metric.to_lowercase().as_str() {
        "cosine" => DistanceMetric::Cosine,
        "euclidean" => DistanceMetric::Euclidean,
        "dotproduct" => DistanceMetric::DotProduct,
        _ => DistanceMetric::Cosine,
    }
}

pub fn vector_record_to_pinecone_vector(
    record: &VectorRecord,
) -> Result<Vector, VectorError> {
    let (values, sparse_values) = match &record.vector {
        VectorData::Dense(dense) => (Some(dense.clone()), None),
        VectorData::Sparse(_) => {
            (Some(vector_data_to_dense(&record.vector)), vector_data_to_sparse(&record.vector))
        }
        VectorData::Hybrid((dense, _sparse)) => {
            (Some(dense.clone()), vector_data_to_sparse(&record.vector))
        }
        _ => {
            (Some(vector_data_to_dense(&record.vector)), None)
        }
    };

    let metadata = record
        .metadata
        .as_ref()
        .map(|m| metadata_to_json_map(m))
        .transpose()?;

    Ok(Vector {
        id: record.id.clone(),
        values,
        sparse_values,
        metadata,
    })
}

pub fn pinecone_vector_to_vector_record(vector: &Vector) -> Result<VectorRecord, VectorError> {
    let vector_data = dense_to_vector_data(vector.values.clone().unwrap_or_default());
    let metadata = vector
        .metadata
        .as_ref()
        .map(|m| json_map_to_metadata(m))
        .transpose()?;

    Ok(VectorRecord {
        id: vector.id.clone(),
        vector: vector_data,
        metadata,
    })
}

pub fn scored_vector_to_search_result(scored: &ScoredVector) -> SearchResult {
    let vector_data = scored.values.as_ref().map(|v| VectorData::Dense(v.clone()));

    let metadata = scored.metadata.as_ref().map(|m| {
        m.iter()
            .map(|(k, v)| {
                let metadata_value = match v {
                    serde_json::Value::String(s) => MetadataValue::StringVal(s.clone()),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            MetadataValue::IntegerVal(i)
                        } else if let Some(f) = n.as_f64() {
                            MetadataValue::NumberVal(f)
                        } else {
                            MetadataValue::NullVal
                        }
                    }
                    serde_json::Value::Bool(b) => MetadataValue::BooleanVal(*b),
                    serde_json::Value::Null => MetadataValue::NullVal,
                    _ => MetadataValue::NullVal,
                };
                (k.clone(), metadata_value)
            })
            .collect()
    });

    SearchResult {
        id: scored.id.clone(),
        score: scored.score,
        distance: 1.0 - scored.score,
        vector: vector_data,
        metadata,
    }
}

pub fn extract_dense_and_sparse_from_query(query: &SearchQuery) -> (Option<Vec<f32>>, Option<crate::client::SparseValues>) {
    match query {
        SearchQuery::Vector(vector_data) => {
            let dense = match vector_data {
                VectorData::Dense(_) | VectorData::Hybrid(_) => Some(vector_data_to_dense(vector_data)),
                _ => None,
            };
            let sparse = vector_data_to_sparse(vector_data);
            (dense, sparse)
        }
        SearchQuery::ById(_) => (None, None),
        SearchQuery::MultiVector(vectors) => {
            if let Some((_, vector_data)) = vectors.first() {
                let dense = match vector_data {
                    VectorData::Dense(_) | VectorData::Hybrid(_) => Some(vector_data_to_dense(vector_data)),
                    _ => None,
                };
                let sparse = vector_data_to_sparse(vector_data);
                (dense, sparse)
            } else {
                (None, None)
            }
        }
    }
}

pub fn pinecone_query_response_to_search_results(
    response: QueryResponse,
) -> Vec<SearchResult> {
    response
        .matches
        .iter()
        .map(scored_vector_to_search_result)
        .collect()
}

pub fn vector_records_to_upsert_request(
    vectors: &[VectorRecord],
    namespace: Option<String>,
) -> Result<UpsertRequest, VectorError> {
    let pinecone_vectors: Result<Vec<Vector>, VectorError> = vectors
        .iter()
        .map(vector_record_to_pinecone_vector)
        .collect();

    Ok(UpsertRequest {
        vectors: pinecone_vectors?,
        namespace,
    })
}

pub fn index_model_to_collection_info(
    model: &IndexModel,
) -> Result<CollectionInfo, VectorError> {
    let metric = string_to_distance_metric(&model.metric);
    let dimension = model.dimension;

   
    let vector_count = 0u64;
    let size_bytes = 0u64; 

    let index_ready = model.status.ready;

    Ok(CollectionInfo {
        name: model.name.clone(),
        description: None,
        dimension,
        metric,
        vector_count,
        size_bytes: Some(size_bytes),
        index_ready,
        created_at: None,
        updated_at: None, 
        provider_stats: None,
    })
}

pub fn pinecone_error_to_vector_error(error: &str) -> VectorError {
    if error.contains("not found") || error.contains("404") {
        VectorError::NotFound(error.to_string())
    } else if error.contains("already exists") || error.contains("409") {
        VectorError::AlreadyExists(error.to_string())
    } else if error.contains("invalid") || error.contains("400") {
        VectorError::InvalidParams(error.to_string())
    } else if error.contains("unauthorized") || error.contains("401") {
        VectorError::Unauthorized(error.to_string())
    } else if error.contains("rate limit") || error.contains("429") {
        VectorError::RateLimited(error.to_string())
    } else if error.contains("dimension") {
        VectorError::DimensionMismatch(error.to_string())
    } else if error.contains("connection") || error.contains("timeout") {
        VectorError::ConnectionError(error.to_string())
    } else {
        VectorError::ProviderError(error.to_string())
    }
}

pub fn filter_expression_to_pinecone_filter(
    filter: &golem_vector::golem::vector::types::FilterExpression,
) -> Result<HashMap<String, serde_json::Value>, VectorError> {
    match filter {
        golem_vector::golem::vector::types::FilterExpression::Condition(condition) => {
            condition_to_pinecone_filter(condition)
        }
        golem_vector::golem::vector::types::FilterExpression::And(filters) => {
            let mut combined_filter = HashMap::new();
            for filter_func in filters {
                let sub_filter = filter_expression_to_pinecone_filter(&filter_func.get())?;
                for (key, value) in sub_filter {
                    combined_filter.insert(key, value);
                }
            }
            Ok(combined_filter)
        }
        golem_vector::golem::vector::types::FilterExpression::Or(filters) => {
            let mut or_conditions = Vec::new();
            for filter_func in filters {
                let sub_filter = filter_expression_to_pinecone_filter(&filter_func.get())?;
                or_conditions.push(serde_json::Value::Object(
                    sub_filter.into_iter().map(|(k, v)| (k, v)).collect()
                ));
            }
            let mut result = HashMap::new();
            result.insert("$or".to_string(), serde_json::Value::Array(or_conditions));
            Ok(result)
        }
        golem_vector::golem::vector::types::FilterExpression::Not(filter_func) => {
            let sub_filter = filter_expression_to_pinecone_filter(&filter_func.get())?;
            let mut result = HashMap::new();
            result.insert(
                "$not".to_string(),
                serde_json::Value::Object(
                    sub_filter.into_iter().map(|(k, v)| (k, v)).collect()
                ),
            );
            Ok(result)
        }
    }
}

pub fn condition_to_pinecone_filter(
    condition: &golem_vector::golem::vector::types::FilterCondition,
) -> Result<HashMap<String, serde_json::Value>, VectorError> {
    use golem_vector::golem::vector::types::{FilterOperator, MetadataValue};
    
    let mut filter = HashMap::new();
    let field = &condition.field;
    
    let pinecone_value = match &condition.value {
        MetadataValue::StringVal(s) => serde_json::Value::String(s.clone()),
        MetadataValue::NumberVal(n) => serde_json::json!(n),
        MetadataValue::IntegerVal(i) => serde_json::json!(i),
        MetadataValue::BooleanVal(b) => serde_json::Value::Bool(*b),
        MetadataValue::NullVal => serde_json::Value::Null,
        _ => {
            return Err(VectorError::UnsupportedFeature(
                "Complex metadata types not supported in Pinecone filters".to_string(),
            ));
        }
    };
    
    match condition.operator {
        FilterOperator::Eq => {
            filter.insert(field.clone(), json!({ "$eq": pinecone_value }));
        }
        FilterOperator::Ne => {
            filter.insert(field.clone(), json!({ "$ne": pinecone_value }));
        }
        FilterOperator::Gt => {
            filter.insert(field.clone(), json!({ "$gt": pinecone_value }));
        }
        FilterOperator::Gte => {
            filter.insert(field.clone(), json!({ "$gte": pinecone_value }));
        }
        FilterOperator::Lt => {
            filter.insert(field.clone(), json!({ "$lt": pinecone_value }));
        }
        FilterOperator::Lte => {
            filter.insert(field.clone(), json!({ "$lte": pinecone_value }));
        }
        FilterOperator::In => {
            if let serde_json::Value::Array(_) = pinecone_value {
                filter.insert(field.clone(), json!({ "$in": pinecone_value }));
            } else {
                return Err(VectorError::InvalidParams(
                    "IN operator requires array value".to_string(),
                ));
            }
        }
        FilterOperator::Nin => {
            if let serde_json::Value::Array(_) = pinecone_value {
                filter.insert(field.clone(), json!({ "$nin": pinecone_value }));
            } else {
                return Err(VectorError::InvalidParams(
                    "NIN operator requires array value".to_string(),
                ));
            }
        }
        _ => {
            return Err(VectorError::UnsupportedFeature(format!(
                "Filter operator {:?} not supported by Pinecone",
                condition.operator
            )));
        }
    }
    
    Ok(filter)
}

pub fn extract_prefix_from_filter(filter: &FilterExpression) -> Option<String> {
        use golem_vector::exports::golem::vector::types::{FilterOperator, MetadataValue};
        
        match filter {
            FilterExpression::Condition(condition) => {
                if condition.field == "id" {
                    match condition.operator {
                        FilterOperator::Contains => {
                            if let MetadataValue::StringVal(prefix) = &condition.value {
                                Some(prefix.clone())
                            } else {
                                None
                            }
                        }
                        _ => None, 
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

#[cfg(test)]
mod tests {
    use super::*;
    use golem_vector::golem::vector::types::{
        SparseVector,
    };
    use serde_json;

    #[test]
    fn test_vector_data_to_dense() {
        let dense_data = vec![1.0, 2.0, 3.0];
        let vector_data = VectorData::Dense(dense_data.clone());
        
        let result = vector_data_to_dense(&vector_data);
        assert_eq!(result, dense_data);
    }

    #[test]
    fn test_vector_data_to_dense_sparse() {
        let sparse = SparseVector {
            indices: vec![0, 2, 4],
            values: vec![1.0, 2.0, 3.0],
            total_dimensions: 5,
        };
        let vector_data = VectorData::Sparse(sparse);
        
        let result = vector_data_to_dense(&vector_data);
        assert_eq!(result, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_metadata_to_json_map() {
        let metadata = vec![
            ("string_field".to_string(), MetadataValue::StringVal("test".to_string())),
            ("number_field".to_string(), MetadataValue::NumberVal(42.5)),
            ("int_field".to_string(), MetadataValue::IntegerVal(100)),
            ("bool_field".to_string(), MetadataValue::BooleanVal(true)),
            ("null_field".to_string(), MetadataValue::NullVal),
        ];

        let result = metadata_to_json_map(&metadata).unwrap();
        
        assert_eq!(result.get("string_field").unwrap(), &serde_json::Value::String("test".to_string()));
        assert_eq!(result.get("number_field").unwrap(), &serde_json::json!(42.5));
        assert_eq!(result.get("int_field").unwrap(), &serde_json::json!(100));
        assert_eq!(result.get("bool_field").unwrap(), &serde_json::Value::Bool(true));
        assert_eq!(result.get("null_field").unwrap(), &serde_json::Value::Null);
    }

    #[test]
    fn test_json_map_to_metadata() {
        let mut map = HashMap::new();
        map.insert("string_field".to_string(), serde_json::Value::String("test".to_string()));
        map.insert("number_field".to_string(), serde_json::json!(42.5));
        map.insert("int_field".to_string(), serde_json::json!(100));
        map.insert("bool_field".to_string(), serde_json::Value::Bool(true));
        map.insert("null_field".to_string(), serde_json::Value::Null);

        let result = json_map_to_metadata(&map).unwrap();
        
        assert_eq!(result.len(), 5);
        assert!(result.contains(&("string_field".to_string(), MetadataValue::StringVal("test".to_string()))));
        assert!(result.contains(&("number_field".to_string(), MetadataValue::NumberVal(42.5))));
        assert!(result.contains(&("int_field".to_string(), MetadataValue::IntegerVal(100))));
        assert!(result.contains(&("bool_field".to_string(), MetadataValue::BooleanVal(true))));
        assert!(result.contains(&("null_field".to_string(), MetadataValue::NullVal)));
    }

    #[test]
    fn test_distance_metric_conversions() {
        assert_eq!(distance_metric_to_string(&DistanceMetric::Cosine), "cosine");
        assert_eq!(distance_metric_to_string(&DistanceMetric::Euclidean), "euclidean");
        assert_eq!(distance_metric_to_string(&DistanceMetric::DotProduct), "dotproduct");

        assert_eq!(string_to_distance_metric("cosine"), DistanceMetric::Cosine);
        assert_eq!(string_to_distance_metric("euclidean"), DistanceMetric::Euclidean);
        assert_eq!(string_to_distance_metric("dotproduct"), DistanceMetric::DotProduct);
        assert_eq!(string_to_distance_metric("unknown"), DistanceMetric::Cosine);
    }

    #[test]
    fn test_vector_record_to_pinecone_vector() {
        let record = VectorRecord {
            id: "test-id".to_string(),
            vector: VectorData::Dense(vec![1.0, 2.0, 3.0]),
            metadata: Some(vec![
                ("category".to_string(), MetadataValue::StringVal("test".to_string())),
            ]),
        };

        let result = vector_record_to_pinecone_vector(&record).unwrap();
        
        assert_eq!(result.id, "test-id");
        assert_eq!(result.values, vec![1.0, 2.0, 3.0].into());
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_pinecone_error_to_vector_error() {
        assert!(matches!(
            pinecone_error_to_vector_error("Index not found"),
            VectorError::NotFound(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Index already exists"),
            VectorError::AlreadyExists(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Invalid parameters"),
            VectorError::InvalidParams(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Unauthorized access"),
            VectorError::Unauthorized(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Rate limit exceeded"),
            VectorError::RateLimited(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Dimension mismatch"),
            VectorError::DimensionMismatch(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Connection timeout"),
            VectorError::ConnectionError(_)
        ));
        
        assert!(matches!(
            pinecone_error_to_vector_error("Unknown error"),
            VectorError::ProviderError(_)
        ));
    }
}

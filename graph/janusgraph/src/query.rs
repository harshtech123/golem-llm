use crate::conversions;
use crate::{GraphJanusGraphComponent, Transaction};
use golem_graph::golem::graph::types::PropertyValue;
use golem_graph::golem::graph::{
    errors::GraphError,
    query::{Guest as QueryGuest, QueryExecutionResult, QueryParameters, QueryResult},
};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;



impl Transaction {

}

impl QueryGuest for GraphJanusGraphComponent {
    fn execute_query(
        transaction: golem_graph::golem::graph::transactions::TransactionBorrow<'_>,
        query: String,
        parameters: Option<QueryParameters>,
        options: Option<golem_graph::golem::graph::query::QueryOptions>,
    ) -> Result<QueryExecutionResult, GraphError> {
        let tx: &Transaction = transaction.get();
        tx.execute_query(query, parameters, options)
    }
}

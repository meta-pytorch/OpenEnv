// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use agentbus_api::write_once_space::WriteOnceError;
use agentbus_api::write_once_space::WriteOnceResult;
use agentbus_api::write_once_space::WriteOnceSpace;
use anyhow::Context;
use anyhow::Result;
use aws_sdk_dynamodb::types::AttributeValue;
use aws_sdk_dynamodb::types::KeySchemaElement;
use aws_sdk_dynamodb::types::KeyType;
use bytes::Bytes;

const SPACE_ID_COLUMN: &str = "space_id";
const ADDRESS_COLUMN: &str = "address";
const VAL_COLUMN: &str = "val";

/// A WriteOnceSpace backed by DynamoDB.
#[derive(Clone)]
pub struct DynamoWriteOnceSpace {
    client: aws_sdk_dynamodb::Client,
    table_name: String,
}

impl DynamoWriteOnceSpace {
    pub async fn new(client: aws_sdk_dynamodb::Client, table_name: String) -> Result<Self> {
        let output = client
            .describe_table()
            .table_name(&table_name)
            .send()
            .await
            .context("failed to describe table")?;

        let table = output.table.context("missing table description")?;
        let key_schema = table.key_schema();

        let has_hash = key_schema.iter().any(|k: &KeySchemaElement| {
            k.attribute_name() == SPACE_ID_COLUMN && k.key_type() == &KeyType::Hash
        });
        let has_range = key_schema.iter().any(|k: &KeySchemaElement| {
            k.attribute_name() == ADDRESS_COLUMN && k.key_type() == &KeyType::Range
        });
        anyhow::ensure!(
            has_hash && has_range,
            "table '{}' must have partition key '{}' (S, HASH) and sort key '{}' (N, RANGE)",
            table_name,
            SPACE_ID_COLUMN,
            ADDRESS_COLUMN,
        );

        Ok(Self { client, table_name })
    }
}

impl WriteOnceSpace for DynamoWriteOnceSpace {
    async fn write(&mut self, space_id: &str, address: u64, value: Bytes) -> WriteOnceResult<()> {
        // TODO: Handle values exceeding the DynamoDB 400KB item size limit.
        let result = self
            .client
            .put_item()
            .table_name(&self.table_name)
            .item(SPACE_ID_COLUMN, AttributeValue::S(space_id.to_string()))
            .item(ADDRESS_COLUMN, AttributeValue::N(address.to_string()))
            .item(
                VAL_COLUMN,
                AttributeValue::B(aws_sdk_dynamodb::primitives::Blob::new(value)),
            )
            .condition_expression(format!(
                "attribute_not_exists({}) AND attribute_not_exists({})",
                SPACE_ID_COLUMN, ADDRESS_COLUMN
            ))
            .send()
            .await;

        match result {
            Ok(_) => Ok(()),
            Err(err) => {
                let service_err = err.into_service_error();
                if service_err.is_conditional_check_failed_exception() {
                    Err(WriteOnceError::AddressAlreadyExists(address))
                } else {
                    Err(WriteOnceError::BackendUnavailable(service_err.to_string()))
                }
            }
        }
    }

    async fn read(&self, space_id: &str, address: u64) -> Option<Bytes> {
        let result = self
            .client
            .get_item()
            .table_name(&self.table_name)
            .key(SPACE_ID_COLUMN, AttributeValue::S(space_id.to_string()))
            .key(ADDRESS_COLUMN, AttributeValue::N(address.to_string()))
            .consistent_read(true)
            .send()
            .await
            .ok()?;

        let item = result.item()?;
        let val = item.get(VAL_COLUMN)?;
        match val {
            AttributeValue::B(blob) => Some(Bytes::copy_from_slice(blob.as_ref())),
            _ => None,
        }
    }

    async fn tail(&self, _space_id: &str) -> WriteOnceResult<u64> {
        Err(WriteOnceError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use aws_sdk_dynamodb::operation::describe_table::DescribeTableOutput;
    use aws_sdk_dynamodb::types::KeySchemaElement;
    use aws_sdk_dynamodb::types::KeyType;
    use aws_sdk_dynamodb::types::TableDescription;
    use aws_smithy_mocks::MockResponseInterceptor;
    use aws_smithy_mocks::Rule;
    use aws_smithy_mocks::RuleMode;
    use aws_smithy_mocks::create_mock_http_client;
    use aws_smithy_mocks::mock;

    use super::*;

    fn mock_client(rules: &[&Rule]) -> aws_sdk_dynamodb::Client {
        let mut interceptor = MockResponseInterceptor::new().rule_mode(RuleMode::Sequential);
        for rule in rules {
            interceptor = interceptor.with_rule(rule);
        }
        aws_sdk_dynamodb::Client::from_conf(
            aws_sdk_dynamodb::Config::builder()
                .region(aws_sdk_dynamodb::config::Region::from_static("us-east-1"))
                .credentials_provider(aws_sdk_dynamodb::config::Credentials::new(
                    "test", "test", None, None, "test",
                ))
                .http_client(create_mock_http_client())
                .interceptor(interceptor)
                .behavior_version_latest()
                .build(),
        )
    }

    fn valid_table_description() -> TableDescription {
        TableDescription::builder()
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(SPACE_ID_COLUMN)
                    .key_type(KeyType::Hash)
                    .build()
                    .unwrap(),
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(ADDRESS_COLUMN)
                    .key_type(KeyType::Range)
                    .build()
                    .unwrap(),
            )
            .build()
    }

    #[tokio::test]
    async fn test_new_accepts_valid_schema() {
        let rule = mock!(aws_sdk_dynamodb::Client::describe_table).then_output(|| {
            DescribeTableOutput::builder()
                .table(valid_table_description())
                .build()
        });
        let client = mock_client(&[&rule]);

        DynamoWriteOnceSpace::new(client, "test-table".to_string())
            .await
            .expect("valid schema should succeed");
    }

    #[tokio::test]
    async fn test_new_rejects_missing_sort_key() {
        let rule = mock!(aws_sdk_dynamodb::Client::describe_table).then_output(|| {
            DescribeTableOutput::builder()
                .table(
                    TableDescription::builder()
                        .key_schema(
                            KeySchemaElement::builder()
                                .attribute_name(SPACE_ID_COLUMN)
                                .key_type(KeyType::Hash)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .build()
        });
        let client = mock_client(&[&rule]);

        let err = DynamoWriteOnceSpace::new(client, "test-table".to_string())
            .await
            .err()
            .expect("should fail with missing sort key");
        assert!(err.to_string().contains("must have partition key"));
    }

    #[tokio::test]
    async fn test_new_rejects_wrong_partition_key_name() {
        let rule = mock!(aws_sdk_dynamodb::Client::describe_table).then_output(|| {
            DescribeTableOutput::builder()
                .table(
                    TableDescription::builder()
                        .key_schema(
                            KeySchemaElement::builder()
                                .attribute_name("wrong_name")
                                .key_type(KeyType::Hash)
                                .build()
                                .unwrap(),
                        )
                        .key_schema(
                            KeySchemaElement::builder()
                                .attribute_name(ADDRESS_COLUMN)
                                .key_type(KeyType::Range)
                                .build()
                                .unwrap(),
                        )
                        .build(),
                )
                .build()
        });
        let client = mock_client(&[&rule]);

        let result = DynamoWriteOnceSpace::new(client, "test-table".to_string()).await;
        assert!(result.is_err(), "should fail with wrong partition key name");
    }

    #[tokio::test]
    async fn test_new_rejects_nonexistent_table() {
        let rule = mock!(aws_sdk_dynamodb::Client::describe_table).then_error(|| {
            aws_sdk_dynamodb::operation::describe_table::DescribeTableError::ResourceNotFoundException(
                aws_sdk_dynamodb::types::error::ResourceNotFoundException::builder()
                    .message("Table not found")
                    .build(),
            )
        });
        let client = mock_client(&[&rule]);

        let err = DynamoWriteOnceSpace::new(client, "test-table".to_string())
            .await
            .err()
            .expect("should fail with nonexistent table");
        assert!(err.to_string().contains("failed to describe table"));
    }
}

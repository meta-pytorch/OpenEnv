// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

use std::rc::Rc;
use std::time::Duration;

use agentbus_api::environment::RealEnvironment;
use agentbus_dynamodb::DynamoWriteOnceSpace;
use agentbus_tests::fixtures::IntegrationFixture;
use agentbus_tests::write_once_space::fixtures::WriteOnceSpaceTestFixture;
use anyhow::Result;
use aws_sdk_dynamodb::types::AttributeDefinition;
use aws_sdk_dynamodb::types::KeySchemaElement;
use aws_sdk_dynamodb::types::KeyType;
use aws_sdk_dynamodb::types::ProvisionedThroughput;
use aws_sdk_dynamodb::types::ScalarAttributeType;
use fbinit::FacebookInit;
use testcontainers::ContainerAsync;
use testcontainers::GenericImage;
use testcontainers::ImageExt;
use testcontainers::core::ContainerPort;
use testcontainers::runners::AsyncRunner;

const TABLE_NAME: &str = "agentbus_test";
const SPACE_ID_COLUMN: &str = "space_id";
const ADDRESS_COLUMN: &str = "address";
const DYNAMODB_PORT: u16 = 8000;

pub struct DynamoWriteOnceSpaceFixture {
    space: DynamoWriteOnceSpace,
    env: Rc<RealEnvironment>,
    _container: ContainerAsync<GenericImage>,
}

impl WriteOnceSpaceTestFixture for DynamoWriteOnceSpaceFixture {
    type Env = RealEnvironment;
    type WriteOnceSpaceImpl = DynamoWriteOnceSpace;

    fn get_env(&self) -> Rc<Self::Env> {
        self.env.clone()
    }

    fn create_impl(&self) -> Self::WriteOnceSpaceImpl {
        self.space.clone()
    }
}

impl DynamoWriteOnceSpaceFixture {
    pub async fn new_async(_fb: FacebookInit) -> Result<Self> {
        let env = Rc::new(RealEnvironment::new());

        let use_host_network = std::env::var("HOST_NETWORK").is_ok();

        let container = if use_host_network {
            GenericImage::new("amazon/dynamodb-local", "latest")
                .with_host_config_modifier(|host_config| {
                    host_config.network_mode = Some("host".to_string());
                })
                .start()
                .await?
        } else {
            GenericImage::new("amazon/dynamodb-local", "latest")
                .with_exposed_port(ContainerPort::Tcp(DYNAMODB_PORT))
                .start()
                .await?
        };

        let endpoint_url = if use_host_network {
            format!("http://127.0.0.1:{}", DYNAMODB_PORT)
        } else {
            let host = container.get_host().await?;
            let port = container.get_host_port_ipv4(DYNAMODB_PORT).await?;
            format!("http://{}:{}", host, port)
        };

        let config = aws_sdk_dynamodb::config::Builder::new()
            .endpoint_url(&endpoint_url)
            .region(aws_sdk_dynamodb::config::Region::new("us-east-1"))
            .credentials_provider(aws_sdk_dynamodb::config::Credentials::new(
                "dummy", "dummy", None, None, "test",
            ))
            .behavior_version_latest()
            .build();

        let client = aws_sdk_dynamodb::Client::from_conf(config);

        // Poll until DynamoDB Local is ready to accept connections.
        for _ in 0..30 {
            if client.list_tables().send().await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }

        client
            .create_table()
            .table_name(TABLE_NAME)
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(SPACE_ID_COLUMN)
                    .key_type(KeyType::Hash)
                    .build()?,
            )
            .key_schema(
                KeySchemaElement::builder()
                    .attribute_name(ADDRESS_COLUMN)
                    .key_type(KeyType::Range)
                    .build()?,
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name(SPACE_ID_COLUMN)
                    .attribute_type(ScalarAttributeType::S)
                    .build()?,
            )
            .attribute_definitions(
                AttributeDefinition::builder()
                    .attribute_name(ADDRESS_COLUMN)
                    .attribute_type(ScalarAttributeType::N)
                    .build()?,
            )
            .provisioned_throughput(
                ProvisionedThroughput::builder()
                    .read_capacity_units(5)
                    .write_capacity_units(5)
                    .build()?,
            )
            .send()
            .await?;

        let space = DynamoWriteOnceSpace::new(client, TABLE_NAME.to_string()).await?;
        Ok(Self {
            space,
            env,
            _container: container,
        })
    }
}

impl IntegrationFixture for DynamoWriteOnceSpaceFixture {
    async fn new_async(fb: FacebookInit) -> Result<Self> {
        Self::new_async(fb).await
    }
}

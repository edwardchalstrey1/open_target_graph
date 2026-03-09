import pulumi
from pulumi_kubernetes.apps.v1 import Deployment, DeploymentSpecArgs
from pulumi_kubernetes.core.v1 import (
    ConfigMap,
    ContainerArgs,
    EnvVarArgs,
    PodSpecArgs,
    PodTemplateSpecArgs,
    Service,
    ServicePortArgs,
    ServiceSpecArgs,
    Secret,
)
from pulumi_kubernetes.meta.v1 import LabelSelectorArgs, ObjectMetaArgs

# Common labels
labels = {"app": "open-target-graph"}

# 1. Postgres Database
postgres_labels = {**labels, "component": "postgres"}
postgres_deployment = Deployment(
    "postgres-deploy",
    spec=DeploymentSpecArgs(
        selector=LabelSelectorArgs(match_labels=postgres_labels),
        replicas=1,
        template=PodTemplateSpecArgs(
            metadata=ObjectMetaArgs(labels=postgres_labels),
            spec=PodSpecArgs(
                containers=[
                    ContainerArgs(
                        name="postgres",
                        image="pgvector/pgvector:pg16",
                        ports=[{"container_port": 5432}],
                        env=[
                            EnvVarArgs(name="POSTGRES_USER", value="admin"),
                            EnvVarArgs(name="POSTGRES_PASSWORD", value="password"),
                            EnvVarArgs(name="POSTGRES_DB", value="open_target_graph"),
                        ],
                    )
                ]
            ),
        ),
    ),
)

postgres_service = Service(
    "postgres-svc",
    metadata=ObjectMetaArgs(name="postgres", labels=postgres_labels),
    spec=ServiceSpecArgs(
        selector=postgres_labels,
        ports=[ServicePortArgs(port=5432, target_port=5432)],
    ),
)

# 2. Dagster Service
dagster_labels = {**labels, "component": "dagster"}
dagster_deployment = Deployment(
    "dagster-deploy",
    spec=DeploymentSpecArgs(
        selector=LabelSelectorArgs(match_labels=dagster_labels),
        replicas=1,
        template=PodTemplateSpecArgs(
            metadata=ObjectMetaArgs(labels=dagster_labels),
            spec=PodSpecArgs(
                containers=[
                    ContainerArgs(
                        name="dagster",
                        image="edchalstrey/open-target-graph-dagster:latest",  # Placeholder
                        ports=[{"container_port": 3000}],
                        env=[
                            EnvVarArgs(name="DAGSTER_HOME", value="/opt/dagster/dagster_home"),
                            EnvVarArgs(name="DB_USER", value="admin"),
                            EnvVarArgs(name="DB_PASSWORD", value="password"),
                            EnvVarArgs(name="DB_HOST", value="postgres"),
                            EnvVarArgs(name="DB_PORT", value="5432"),
                            EnvVarArgs(name="DB_NAME", value="open_target_graph"),
                        ],
                    )
                ]
            ),
        ),
    ),
)

dagster_service = Service(
    "dagster-svc",
    metadata=ObjectMetaArgs(name="dagster", labels=dagster_labels),
    spec=ServiceSpecArgs(
        selector=dagster_labels,
        ports=[ServicePortArgs(port=3000, target_port=3000)],
    ),
)

# 3. Streamlit Dashboard
streamlit_labels = {**labels, "component": "streamlit"}
streamlit_deployment = Deployment(
    "streamlit-deploy",
    spec=DeploymentSpecArgs(
        selector=LabelSelectorArgs(match_labels=streamlit_labels),
        replicas=1,
        template=PodTemplateSpecArgs(
            metadata=ObjectMetaArgs(labels=streamlit_labels),
            spec=PodSpecArgs(
                containers=[
                    ContainerArgs(
                        name="streamlit",
                        image="edchalstrey/open-target-graph-streamlit:latest",  # Placeholder
                        ports=[{"container_port": 8501}],
                        env=[
                            EnvVarArgs(name="DB_USER", value="admin"),
                            EnvVarArgs(name="DB_PASSWORD", value="password"),
                            EnvVarArgs(name="DB_HOST", value="postgres"),
                            EnvVarArgs(name="DB_PORT", value="5432"),
                            EnvVarArgs(name="DB_NAME", value="open_target_graph"),
                            # GEMINI_API_KEY should be a Secret
                            EnvVarArgs(name="GEMINI_API_KEY", value_from={"secret_key_ref": {"name": "api-secrets", "key": "GEMINI_API_KEY"}}),
                        ],
                    )
                ]
            ),
        ),
    ),
)

streamlit_service = Service(
    "streamlit-svc",
    metadata=ObjectMetaArgs(name="streamlit", labels=streamlit_labels),
    spec=ServiceSpecArgs(
        selector=streamlit_labels,
        ports=[ServicePortArgs(port=8501, target_port=8501)],
    ),
)

# Exports
pulumi.export("dagster_service_name", dagster_service.metadata["name"])
pulumi.export("streamlit_service_name", streamlit_service.metadata["name"])

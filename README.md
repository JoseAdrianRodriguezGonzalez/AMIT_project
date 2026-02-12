flowchart TD

    A[Inicio main()] --> B[Cargar archivos desde ../data]
    B --> C[data_loader(chol)]
    C --> D[datasets = loader.load()]

    D --> E{Iterar sobre datasets}
    
    E --> F[Inicializar Preprocessor]
    F --> G[Inicializar Vectorizer]
    
    G --> H[docs = pre.transform(df)]
    H --> I[NERextractor]
    I --> J[entities = ner.extract_entities(docs)]
    J --> K[ner.top_entities() + get_top_entities()]
    
    K --> L[X, vocab = vec.build_vocab(docs)]
    L --> M[vector = vec.get_topic_vectorizer()]
    M --> N[TopicModeler(vector)]
    
    N --> O[TopicNERAnalyzer(tp, ner)]
    O --> P[topic_entity_summary = analyzer.fit(docs)]
    
    P --> Q{¿Hay tópicos?}
    Q -- Sí --> R[visualize_topic_entities()]
    Q -- No --> S[Continuar]

    R --> T[tp.fit(docs)]
    S --> T

    T --> U[Obtener freq_info]
    U --> V[Mostrar topics y probabilidades]
    
    V --> W[Seleccionar modelo LLM]
    W --> X[LlamaModel(path_model)]
    
    X --> Y[request_title(freq['Representation'])]
    Y --> Z[Imprimir títulos descriptivos]
    
    Z --> AA[request_interpretation(freq['Representation'])]
    AA --> AB[Imprimir interpretaciones]

    AB --> AC[Visualizaciones: barchart]
    AC --> AD[Visualizaciones: hierarchy]
    AD --> AE[Visualizaciones: topics]
    
    AE --> AF[Mostrar figuras]
    AF --> AG[Break loop]
    AG --> AH[Fin]

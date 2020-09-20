# Klay - Clean DL4J Network Declarations

This project, Klay (Kotlin LAYers), uses Kotlin to build a Domain-Specific Language (DSL) for defining neural networks in DL4J.
It is accompanied by [this blog post]().

Klay uses type-safe builder functions to initialize layers of neural networks.
Instead of writing this in standard DL4J syntax:

```kotlin
val denseLayer = DenseLayer.Builder()
                    .nIn(10)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .build()
```

We can use this more concise form:

```kotlin
val denseLayer = dense {
                        nIn(10)
                        nOut(100)
                        activation(Activation.RELU)
                       }
```

For more information, please refere to the blog post.

This repository contains all quickstart modelling examples from the DL4J example repository.
They were converted to Kotlin and show the capabilities of Klay.
Only the layers necessary for getting the examples to work are implemented.
The rest is still work in progress.
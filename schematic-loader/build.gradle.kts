plugins {
    id("application")
}

group = "org.text2mc"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    implementation("net.sandrohc:schematic4j:1.1.0")
    implementation("commons-io:commons-io:2.15.1")
    implementation("com.google.code.gson:gson:2.10.1")
}

buildscript {
    repositories {
        maven {
            url = uri("https://plugins.gradle.org/m2/")
        }
    }
    dependencies {
        classpath("com.github.johnrengelman:shadow:8.1.1")
    }
}

apply(plugin = "com.github.johnrengelman.shadow")

tasks.test {
    useJUnitPlatform()
}

application {
    mainClass = "org.text2mc.Main"
}
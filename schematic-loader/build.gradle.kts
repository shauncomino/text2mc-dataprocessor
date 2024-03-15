plugins {
    id("java")
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
}

tasks.test {
    useJUnitPlatform()
}
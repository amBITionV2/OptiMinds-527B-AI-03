package com.saharsh.Code.editor.Platform.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig {

    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                        // exact localhost ports
                        .allowedOrigins(
                                "http://localhost:3000",
                                "http://localhost:5173",
                                "http://localhost:5174",
                                "http://localhost:5175",
                                "http://127.0.0.1:5173",
                                "http://127.0.0.1:5174",
                                "http://127.0.0.1:5175"
                        )
                        // and patterns for devtunnels / other localhost ports
                        .allowedOriginPatterns(
                                "http://127.0.0.1:*",
                                "http://localhost:*",
                                "https://*.inc1.devtunnels.ms"
                        )
                        .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                        .allowedHeaders("*")
                        .exposedHeaders("*")
                        .allowCredentials(true)  // only if you need cookies/Authorization to be sent
                        .maxAge(3600);
            }
        };
    }
}

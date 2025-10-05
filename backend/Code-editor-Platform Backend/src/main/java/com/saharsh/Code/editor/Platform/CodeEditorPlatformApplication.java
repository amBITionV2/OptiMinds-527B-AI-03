package com.saharsh.Code.editor.Platform;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

@SpringBootApplication
public class CodeEditorPlatformApplication {
	public static void main(String[] args) {
		ApplicationContext context =  SpringApplication.run(CodeEditorPlatformApplication.class, args);
	}
}

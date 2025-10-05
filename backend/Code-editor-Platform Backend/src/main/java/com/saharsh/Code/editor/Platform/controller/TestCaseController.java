package com.saharsh.Code.editor.Platform.controller;

import com.saharsh.Code.editor.Platform.model.TestCase;
import com.saharsh.Code.editor.Platform.service.TestCaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/testcases")
public class TestCaseController {

    @Autowired
    private TestCaseService testCaseService;

    // Get all test cases
    @GetMapping
    public ResponseEntity<List<TestCase>> getAllTestCases() {
        List<TestCase> testCases = testCaseService.getAllTestCases();
        return new ResponseEntity<>(testCases, HttpStatus.OK);
    }

    // Get test case by ID
    @GetMapping("/{id}")
    public ResponseEntity<TestCase> getTestCaseById(@PathVariable int id) {
        Optional<TestCase> testCase = testCaseService.getTestCaseById(id);
        return testCase.map(value -> new ResponseEntity<>(value, HttpStatus.OK))
                .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }


    @GetMapping("/question/{questionId}")
    public ResponseEntity<TestCase> getFirstTestCaseByQuestionId(@PathVariable int questionId) {
        Optional<TestCase> testCase = testCaseService.getFirstTestCaseByQuestionId(questionId);
        return testCase.map(tc -> new ResponseEntity<>(tc, HttpStatus.OK))
                .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }


    // Create a new test case
    @PostMapping
    public ResponseEntity<TestCase> createTestCase(@RequestBody TestCase testCase) {

        TestCase savedTestCase = testCaseService.saveTestCase(testCase);
        return new ResponseEntity<>(savedTestCase, HttpStatus.CREATED);
    }

    // Update an existing test case
    @PutMapping("/{id}")
    public ResponseEntity<TestCase> updateTestCase(@PathVariable int id, @RequestBody TestCase testCase) {
        Optional<TestCase> existingTestCase = testCaseService.getTestCaseById(id);
        if (existingTestCase.isPresent()) {
            testCase.setTestCaseId(id); // Ensure the ID from the path is used
            TestCase updatedTestCase = testCaseService.saveTestCase(testCase);
            return new ResponseEntity<>(updatedTestCase, HttpStatus.OK);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    // Delete a test case by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTestCase(@PathVariable int id) {
        if (testCaseService.getTestCaseById(id).isPresent()) {
            testCaseService.deleteTestCase(id);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }
}
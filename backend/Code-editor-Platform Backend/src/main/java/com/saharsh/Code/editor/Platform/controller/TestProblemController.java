package com.saharsh.Code.editor.Platform.controller;

import com.saharsh.Code.editor.Platform.model.TestProblem;
import com.saharsh.Code.editor.Platform.service.TestProblemService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/testproblems")
public class TestProblemController {

    @Autowired
    private TestProblemService testProblemService;

    // Get all test problems
    @GetMapping
    public ResponseEntity<List<TestProblem>> getAllTestProblems() {
        List<TestProblem> testProblems = testProblemService.getAllTestProblems();
        return new ResponseEntity<>(testProblems, HttpStatus.OK);
    }

    // Get test problem by ID
    @GetMapping("/{id}")
    public ResponseEntity<TestProblem> getTestProblemById(@PathVariable int id) {
        Optional<TestProblem> testProblem = testProblemService.getTestProblemById(id);
        return testProblem.map(value -> new ResponseEntity<>(value, HttpStatus.OK))
                .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }

    // Create a new test problem
    @PostMapping
    public ResponseEntity<TestProblem> createTestProblem(@RequestBody TestProblem testProblem) {
        TestProblem savedTestProblem = testProblemService.saveTestProblem(testProblem);
        return new ResponseEntity<>(savedTestProblem, HttpStatus.CREATED);
    }

    // Update an existing test problem
    @PutMapping("/{id}")
    public ResponseEntity<TestProblem> updateTestProblem(@PathVariable int id, @RequestBody TestProblem testProblem) {
        Optional<TestProblem> existingTestProblem = testProblemService.getTestProblemById(id);
        if (existingTestProblem.isPresent()) {
            testProblem.setId(id); // Ensure the ID from the path is used
            TestProblem updatedTestProblem = testProblemService.saveTestProblem(testProblem);
            return new ResponseEntity<>(updatedTestProblem, HttpStatus.OK);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    // Delete a test problem by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteTestProblem(@PathVariable int id) {
        if (testProblemService.getTestProblemById(id).isPresent()) {
            testProblemService.deleteTestProblem(id);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }
}
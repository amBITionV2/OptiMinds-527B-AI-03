package com.saharsh.Code.editor.Platform.controller;

import com.saharsh.Code.editor.Platform.model.Submission;
import com.saharsh.Code.editor.Platform.service.SubmissionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Optional;

@RestController
@RequestMapping("/api/submissions")
public class SubmissionController {

    @Autowired
    private SubmissionService submissionService;

    // Get all submissions
    @GetMapping
    public ResponseEntity<List<Submission>> getAllSubmissions() {
        List<Submission> submissions = submissionService.getAllSubmissions();
        return new ResponseEntity<>(submissions, HttpStatus.OK);
    }

    // Get submission by ID
    @GetMapping("/{id}")
    public ResponseEntity<Submission> getSubmissionById(@PathVariable int id) {
        Optional<Submission> submission = submissionService.getSubmissionById(id);
        return submission.map(value -> new ResponseEntity<>(value, HttpStatus.OK))
                .orElseGet(() -> new ResponseEntity<>(HttpStatus.NOT_FOUND));
    }

    // Create a new submission
    @PostMapping
    public ResponseEntity<Submission> createSubmission(@RequestBody Submission submission) {
        Submission savedSubmission = submissionService.saveSubmission(submission);
        return new ResponseEntity<>(savedSubmission, HttpStatus.CREATED);
    }

    // Update an existing submission
    @PutMapping("/{id}")
    public ResponseEntity<Submission> updateSubmission(@PathVariable int id, @RequestBody Submission submission) {
        Optional<Submission> existingSubmission = submissionService.getSubmissionById(id);
        if (existingSubmission.isPresent()) {
            submission.setSubmissionId(id); // Ensure the ID from the path is used
            Submission updatedSubmission = submissionService.saveSubmission(submission);
            return new ResponseEntity<>(updatedSubmission, HttpStatus.OK);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    // Delete a submission by ID
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteSubmission(@PathVariable int id) {
        if (submissionService.getSubmissionById(id).isPresent()) {
            submissionService.deleteSubmission(id);
            return new ResponseEntity<>(HttpStatus.NO_CONTENT);
        } else {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }
}
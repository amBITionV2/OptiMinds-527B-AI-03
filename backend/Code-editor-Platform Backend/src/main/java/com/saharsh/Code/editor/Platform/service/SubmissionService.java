package com.saharsh.Code.editor.Platform.service;

import com.saharsh.Code.editor.Platform.model.Submission;
import com.saharsh.Code.editor.Platform.repo.SubmissionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class SubmissionService {

    @Autowired
    private SubmissionRepository submissionRepository;

    public List<Submission> getAllSubmissions() {
        return submissionRepository.findAll();
    }

    public Optional<Submission> getSubmissionById(int id) {
        return submissionRepository.findById(id);
    }

    public Submission saveSubmission(Submission submission) {
        return submissionRepository.save(submission);
    }

    public void deleteSubmission(int id) {
        submissionRepository.deleteById(id);
    }
}

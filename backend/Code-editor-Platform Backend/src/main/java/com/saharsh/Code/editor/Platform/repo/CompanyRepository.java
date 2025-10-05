package com.saharsh.Code.editor.Platform.repo;

import com.saharsh.Code.editor.Platform.model.Company;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CompanyRepository extends JpaRepository<Company, Integer> {
    // You can add custom query methods here if needed
}

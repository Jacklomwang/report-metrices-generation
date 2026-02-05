classdef SessionReport < mlreportgen.report.Report 

    properties 
        Age
        Height
        Weight
        Date
        randomtest
        ReactionTime
        congruency
        InhaleVolume
        SpiroTest
        sprio_Table
        Mytable
        Table2
        valfigure
        Valratio
        valvalue
        sympafaliure
        sympafaliureValue
        STSplots
        STSTable
        EIratio
        DPBfig
        DeltaHR
        Table5
    end 

    methods 
        function obj = SessionReport(varargin) 
            obj = obj@mlreportgen.report.Report(varargin{:}); 
        end 
    end 

    methods (Hidden) 
        function templatePath = getDefaultTemplatePath(rpt) 
            path = myReports.SessionReport.getClassFolder(); 
            templatePath = ... 
                mlreportgen.report.ReportForm.getFormTemplatePath(... 
                path, rpt.Type); 
        end 

    end 

    methods (Static) 
        function path = getClassFolder() 
            [path] = fileparts(mfilename('fullpath')); 
        end 

        function createTemplate(templatePath, type) 
            path = myReports.SessionReport.getClassFolder(); 
            mlreportgen.report.ReportForm.createFormTemplate(... 
                templatePath, type, path); 
        end 

        function customizeReport(toClasspath) 
            mlreportgen.report.ReportForm.customizeClass(... 
                toClasspath, "myReports.SessionReport"); 
        end 

    end  
end

Pod::Spec.new do |spec|

  spec.name         = "btbpAILib"
  spec.version      = "0.0.1"
  spec.summary      = "A short description of btbpAILib. Contains BTBP AI Lib Entry points"

  spec.description  = <<-DESC
				A library with BTBP AI simulations n face.
                   DESC

  spec.homepage     = "http://EXAMPLE/btbpAILib"
  
  spec.license      = "MIT"
  
  spec.author             = { "SriDurga" => "107485940+NarasimhaBTBP@users.noreply.github.com" }
 
  spec.source       = { :git => "https://github.com/btbpAILib.git", :tag => "#{spec.version}" }


  spec.source_files  = "Classes", "Classes/**/*.{h,m}"
  spec.exclude_files = "Classes/Exclude"


end

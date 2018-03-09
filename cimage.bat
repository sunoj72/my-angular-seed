@ECHO OFF

IF "%1"=="" GOTO usage
IF "%2"=="" GOTO usage
IF "%3"=="" GOTO usage

@REM RG_Name, VM_Name, Image_Name

@ECHO unprovisoned VM Image needed. (waagent -deprovision+user -force)

cmd /c az vm deallocate --resource-group %1 --name %2
cmd /c az vm generalize --resource-group %1 --name %2
cmd /c az image create --resource-group %1 --name %3 --source %2
GOTO END

:usage
@ECHO create_image.bat RESOURC_GROUP_NAME VM_NAME IMAGE_NAME
@ECHO ON

:END 
